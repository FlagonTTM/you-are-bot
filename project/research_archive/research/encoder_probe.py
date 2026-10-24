"""Frozen H9: offline multilingual encoder plus the entire strong feature bank."""
from __future__ import annotations

import os
# Hard offline policy is set before importing Transformers/Hugging Face.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import save_file, load_file
import experiment
from botbench import file_sha
from research.sequence_probe import Preprocessor, seed_all, synchronize, write_json

CONFIG = dict(version='H9-encoder-v2', model='intfloat/multilingual-e5-small',
              revision='614241f622f53c4eeff9890bdc4f31cfecc418b3', seed=1729,
              epochs=5, batch_size=16, encoder_lr=2e-5, head_lr=3e-4,
              weight_decay=.01, adam_betas=[.9,.999], adam_eps=1e-8, gradnorm=1.,
              cpu_threads=4, mps_memory_fraction=.5, dtype='float32', amp=False,
              max_messages=16, max_characters=500, max_length=384,
              truncation_side='right', padding='dynamic', category_embedding=8,
              numeric_columns=162, category_columns=14, head=128, dropout=.2,
              pooling='attention_mask_mean_then_LayerNorm', attention='eager',
              primary_weight=.5, checkpoint_tolerance=1e-6, resumable=False)


def configure(device):
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('disable PYTORCH_ENABLE_MPS_FALLBACK; no CPU fallback')
    torch.set_num_threads(4)
    if device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no silent fallback')
        torch.mps.set_per_process_memory_fraction(CONFIG['mps_memory_fraction'])
    elif device != 'cpu':
        raise ValueError('explicit cpu or mps required')
    seed_all()
    return device

PREFIX = 'query: Decide whether speaker A is a bot.\n'


def render(row):
    participant = row['participant_index']
    if participant not in (0, 1):
        raise ValueError('invalid actual participant role')
    ordered = sorted(row['messages'], key=lambda m: m['message'])
    if len(ordered) > 16:
        ordered = ordered[:8] + ordered[-8:]
    parts = []
    for message in ordered:
        if message['participant_index'] not in (0, 1):
            raise ValueError('invalid actual message participant role')
        text = experiment.clean_text(message['text'])
        if len(text) > 500:
            text = text[:375] + text[-125:]
        parts.append(('A: ' if message['participant_index'] == participant else 'B: ') + text)
    return PREFIX + '\n'.join(parts)


@dataclass
class Prepared:
    encodings: list
    numeric: np.ndarray
    categories: np.ndarray

    def __len__(self):
        return len(self.encodings)


def prepare(rows, prep, tokenizer, bank=None):
    # Metadata/labels are never included in a network input.
    bank = experiment.model_input(rows, 'template') if bank is None else bank
    numeric, categories = prep.transform_bank(bank)
    if numeric.shape[1] != 162 or categories.shape[1] != 14:
        raise ValueError('require ENTIRE strong bank: 162 numeric +14 categories')
    tokenizer.truncation_side = 'right'
    tokenizer.padding_side = 'right'
    encoded = tokenizer([render(r) for r in rows], padding=False, truncation=True,
                        max_length=384, add_special_tokens=True, return_attention_mask=True,
                        return_token_type_ids=True)
    encodings = [{k:list(v[i]) for k,v in encoded.items() if k in ('input_ids','attention_mask','token_type_ids')}
                 for i in range(len(rows))]
    return Prepared(encodings, numeric, categories)


def fit_preparation(rows, tokenizer):
    bank = experiment.model_input(rows, 'template')
    prep = Preprocessor().fit_bank(bank)
    return prep, prepare(rows, prep, tokenizer, bank)


def tensor_batch(data, indices, tokenizer, device):
    indices = np.asarray(indices, dtype=np.int64)
    padded = tokenizer.pad([data.encodings[i] for i in indices], padding=True, return_tensors='pt')
    batch = {k:v.to(device=device, dtype=torch.long) for k,v in padded.items()
             if k in ('input_ids','attention_mask','token_type_ids')}
    batch['numeric'] = torch.as_tensor(data.numeric[indices], dtype=torch.float32, device=device)
    batch['categories'] = torch.as_tensor(data.categories[indices], dtype=torch.long, device=device)
    return batch


class EncoderModel(nn.Module):
    def __init__(self, encoder, numeric_dim, category_sizes):
        super().__init__()
        if numeric_dim != 162 or len(category_sizes) != 14:
            raise ValueError('full strong bank schema required')
        self.encoder, self.numeric_dim, self.category_sizes = encoder, numeric_dim, list(category_sizes)
        width = encoder.config.hidden_size
        self.normalization = nn.LayerNorm(width)
        self.category_embeddings = nn.ModuleList([nn.Embedding(n,8) for n in category_sizes])
        self.head = nn.Sequential(nn.Linear(width + numeric_dim + 14*8,128), nn.GELU(),
                                  nn.Dropout(.2), nn.Linear(128,1))

    def forward(self, input_ids, attention_mask, numeric, categories, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = self.normalization((hidden*weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1))
        cats = torch.cat([emb(categories[:,i]) for i,emb in enumerate(self.category_embeddings)],dim=1)
        return self.head(torch.cat([pooled,numeric,cats],dim=1)).squeeze(-1)


def optimizer_for(model):
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    other = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW([dict(params=encoder,lr=2e-5),dict(params=other,lr=3e-4)],
        betas=(.9,.999), eps=1e-8, weight_decay=.01, amsgrad=False, maximize=False,
        foreach=False, capturable=False, differentiable=False, fused=False)


def train_step(model, optimizer, batch, labels, micro_batch_size=16, empty_mps_cache=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    size = len(labels)
    if micro_batch_size not in (1,2,4,8,16):
        raise ValueError('microbatch must divide effective batch 16')
    total = labels.new_zeros(())
    for start in range(0,size,micro_batch_size):
        stop = min(start+micro_batch_size,size)
        micro = {k:v[start:stop] for k,v in batch.items()}
        if 'attention_mask' in micro:
            width = int(micro['attention_mask'].sum(dim=1).max().item())
            for key in ('input_ids','attention_mask','token_type_ids'):
                if key in micro:
                    micro[key] = micro[key][:,:width]
        loss = nn.functional.binary_cross_entropy_with_logits(model(**micro),labels[start:stop])
        loss = loss*((stop-start)/size)
        if not torch.isfinite(loss):
            raise ValueError('non-finite training loss')
        loss.backward()
        total += loss.detach()
        del loss,micro
    if empty_mps_cache and labels.device.type=='mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    norm = nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    if empty_mps_cache and labels.device.type=='mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return total, norm.detach()


def predict_prepared(model, data, tokenizer, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0,len(data),16):
            batch = tensor_batch(data, range(start,min(start+16,len(data))),tokenizer,device)
            parts.append(torch.sigmoid(model(**batch)).cpu().numpy().astype(np.float64))
    return np.concatenate(parts)


def save_bundle(path, model, prep, metadata):
    path = Path(path)
    path.mkdir(parents=True,exist_ok=False)
    temporary = path / 'model.safetensors.partial'
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(temporary))
    temporary.rename(path/'model.safetensors')
    write_json(path/'config.json', model.encoder.config.to_dict())
    write_json(path/'metadata.json',dict(config=CONFIG, numeric_dim=model.numeric_dim,
        category_sizes=model.category_sizes,preprocessing=prep.to_dict(),metadata=metadata,
        weights_sha256=file_sha(path/'model.safetensors'),config_sha256=file_sha(path/'config.json')))


def load_bundle(path, device):
    path = Path(path)
    info = json.loads((path/'metadata.json').read_text())
    if (info['config'] != CONFIG or file_sha(path/'model.safetensors') != info['weights_sha256']
            or file_sha(path/'config.json') != info['config_sha256']):
        raise ValueError('checkpoint config/hash binding mismatch')
    config = AutoConfig.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    model = EncoderModel(AutoModel.from_config(config,trust_remote_code=False,attn_implementation='eager'),
                         info['numeric_dim'],info['category_sizes'])
    model.load_state_dict(load_file(str(path/'model.safetensors'),device='cpu'),strict=True)
    return model.to(device=device,dtype=torch.float32),Preprocessor.from_dict(info['preprocessing']),info['metadata']


from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import socket
from research.sequence_probe import Resources, load_training, split_indices

ROOT = Path(__file__).resolve().parents[1]
OWNED_RUNS = ROOT/'runs/encoder-preflight-v2'
MODEL_DIR = ROOT/'models/multilingual-e5-small'
ASSET_RECEIPT = ROOT/'evidence/encoder-assets-verified.json'
NATIVE = ROOT/'runs/template-v1'
H8 = ROOT/'runs/sequence-full-v2'


@contextmanager
def no_network():
    def denied(*args,**kwargs):
        raise RuntimeError('network disabled in H9 runtime')
    with ExitStack() as stack:
        for target in ('socket.create_connection','socket.getaddrinfo','socket.socket.connect','socket.socket.connect_ex'):
            stack.enter_context(patch(target,denied))
        yield


def load_verified_assets(model_dir=MODEL_DIR, receipt_path=ASSET_RECEIPT):
    # Parent owns assets; no importer/downloader fallback is permitted.
    if Path(model_dir).resolve() != MODEL_DIR or Path(receipt_path).resolve() != ASSET_RECEIPT:
        raise ValueError('only the exact pinned multilingual assets are permitted')
    try:
        from research.encoder_assets import verify_assets
    except ImportError as exc:
        raise RuntimeError('parent asset verifier unfinished; benchmark blocked') from exc
    receipt = verify_assets(Path(model_dir),Path(receipt_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,
        trust_remote_code=False,use_fast=True,truncation_side='right',padding_side='right',model_max_length=384)
    if not tokenizer.is_fast:
        raise ValueError('fast local tokenizer required; no sentencepiece install/download')
    config = AutoConfig.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    if config.model_type != 'bert' or config.architectures != ['BertModel']:
        raise ValueError('pinned actual BertModel architecture mismatch')
    encoder = AutoModel.from_pretrained(str(model_dir),config=config,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='eager')
    return encoder,tokenizer,receipt


def benchmark_batches(data, rows=None):
    lengths = [len(x['input_ids']) for x in data.encodings]
    raw = [sum(len(m['text']) for m in r['messages']) for r in rows] if rows is not None else lengths
    ranked = np.asarray(sorted(range(len(data)),key=lambda i:(lengths[i],raw[i],i)))
    if len(ranked) < 80:
        raise ValueError('five disjoint complete TRAIN batches of 16 required')
    rng = np.random.default_rng(1729)
    batches = [rng.choice(chunk,16,replace=False) for chunk in np.array_split(ranked,5)]
    longest = max(range(len(data)),key=lambda i:(lengths[i],raw[i],i))
    for b in batches[:-1]:
        if longest in b:
            replacement = next(i for i in ranked if i not in np.concatenate(batches))
            b[np.flatnonzero(b == longest)[0]] = replacement
    if longest not in batches[-1]:
        batches[-1][-1] = longest
    return batches,longest


def project_training(train_rows, preparation_seconds, load_seconds, step_seconds):
    steps = int(np.ceil(train_rows/16))*5
    warm = float(np.mean(step_seconds[1:]))
    return dict(steps=steps,train_rows=int(train_rows),epochs=5,
        warm_seconds=float(preparation_seconds+load_seconds+step_seconds[0]+(steps-1)*warm),
        conservative_seconds=float(preparation_seconds+load_seconds+steps*max(step_seconds)),
        excludes='evaluation preparation/inference, checkpoint IO, native replay and bootstrap reports')


def new_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OWNED_RUNS) or path == OWNED_RUNS:
        raise ValueError('output must be a new child of owned runs/encoder-preflight-v2/')
    path.mkdir(parents=True,exist_ok=False)
    return path


class ResourceSampler(Resources):
    def sample(self):
        super().sample()
        import psutil
        times = psutil.Process().cpu_times()
        self.samples[-1]['process_cpu_seconds'] = times.user+times.system
        if len(self.samples)>1:
            a,b = self.samples[-2:]
            b['sampled_cpu_cores'] = (b['process_cpu_seconds']-a['process_cpu_seconds'])/(b['seconds']-a['seconds'])

    def report(self):
        report = super().report()
        report['peak_measurement'] = 'sampled, not allocator high-water mark'
        report['max_sampled_process_cpu_cores'] = max(s.get('sampled_cpu_cores',0.) for s in self.samples)
        return report


def source_closure():
    baseline = json.loads((ROOT/'evidence/guard-correction-v2.json').read_text())['new_verification_source']
    if any(file_sha(ROOT/n)!=h for n,h in baseline.items()):
        raise ValueError('incumbent source changed')
    names = [*baseline,'research/encoder_probe.py','tests/test_encoder_probe.py',
        'research/sequence_probe.py','tests/test_sequence_probe.py','research/encoder_assets.py',
        'requirements.txt','evidence/guard-correction-v2.json','evidence/membership-v2.json',
        'evidence/rules.txt','evidence/encoder-v2-prereg.md','evidence/pretrained-encoder-authorization-v2.json',
        'evidence/encoder-memory-authorization-v2.json',
        'evidence/encoder-assets-verified.json','data/train.json','data/ytrain.csv',
        'runs/template-v1/oof.csv','runs/template-v1/report.json',
        'runs/sequence-full-v2/report.json','runs/sequence-full-v2/primary.csv',
        'runs/sequence-full-v2/sequence.csv','runs/sequence-full-v2/source-closure.json']
    native_report = json.loads((NATIVE/'report.json').read_text())
    # BRIEF records changing campaign/approval status, not a consumed model input.
    # Keep its exact historical snapshot bound; H9 permission is pinned above.
    names += [n for n in native_report['provenance']['sha256'] if n!='BRIEF.md']
    names += [str(p.relative_to(ROOT)) for p in sorted((NATIVE/'snapshot').rglob('*')) if p.is_file()]
    names += [str(p.relative_to(ROOT)) for p in sorted(MODEL_DIR.iterdir()) if p.is_file()]
    names += [str(p.relative_to(ROOT)) for p in sorted(NATIVE.glob('fold-*.joblib'))]
    for fold in range(5):
        names += [f'runs/sequence-full-v2/fold-{fold}/{n}' for n in
                  ('membership.json','sequence/metadata.json','sequence/weights.pt')]
    return {n:file_sha(ROOT/n) for n in dict.fromkeys(names)}


def snapshot(out, closure):
    import shutil
    # Large assets/checkpoints remain read-only in their hash-bound locations.
    for name,digest in closure.items():
        if name.startswith(('models/','runs/')):
            continue
        target = out/'snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        if file_sha(target) != digest:
            raise ValueError('source changed during snapshot')
    write_json(out/'source-closure.json',closure)
    write_json(out/'config.json',CONFIG)


def replay_gate(path, model, data, tokenizer, device, rows=None):
    before = predict_prepared(model,data,tokenizer,device)
    loaded,restored,_ = load_bundle(path,device)
    replay_data = prepare(rows,restored,tokenizer) if rows is not None else data
    after = predict_prepared(loaded,replay_data,tokenizer,device)
    delta = float(np.max(np.abs(before-after)))
    del loaded
    if delta > 1e-6:
        raise ValueError(f'checkpoint replay exceeds preregistered tolerance: {delta}')
    return delta


def benchmark(out, device, encoder, tokenizer, receipt, closure, load_seconds, execution):
    start = time.perf_counter()
    snapshot(out,closure)
    rows,folds = load_training()
    training = [r for r,f in zip(rows,folds) if f>0]
    prep,data = fit_preparation(training,tokenizer)
    batches,longest = benchmark_batches(data,training)
    if len(data.encodings[longest]['input_ids']) != 384:
        raise ValueError('longest real TRAIN case must exercise 384-token limit')
    preparation_seconds = time.perf_counter()-start
    write_json(out/'membership.json',dict(train_IDs=[r['ID'] for r in training],evaluation_IDs=[],
        excluded_folds=[-1,0],batches=[[training[i]['ID'] for i in b] for b in batches],
        longest_ID=training[longest]['ID'],longest_token_count=384,
        longest_case_selection='max token length after masking/clipping; raw length tie-break'))
    initialization_start = time.perf_counter()
    seed_all()
    model = EncoderModel(encoder,prep.numeric_dim,prep.category_sizes).to(device=device,dtype=torch.float32)
    optimizer = optimizer_for(model)
    synchronize(device)
    initialization_seconds = time.perf_counter()-initialization_start
    latencies = []
    batch_shapes = []
    for batch_number,indices in enumerate(batches,1):
        synchronize(device)
        begin = time.perf_counter()
        batch = tensor_batch(data,indices,tokenizer,device)
        labels = torch.as_tensor([training[i]['y'] for i in indices],device=device,dtype=torch.float32)
        train_step(model,optimizer,batch,labels,micro_batch_size=execution['micro_batch_size'],
                   empty_mps_cache=execution['empty_mps_cache'])
        synchronize(device)
        latencies.append(time.perf_counter()-begin)
        batch_shapes.append(list(batch['input_ids'].shape))
        print(json.dumps(dict(train_batch=batch_number,seconds=latencies[-1],shape=batch_shapes[-1],quality_scored=False)),flush=True)
    del optimizer
    model.zero_grad(set_to_none=True)
    if device=='mps':
        torch.mps.empty_cache()
    checkpoint_start = time.perf_counter()
    save_bundle(out/'checkpoint',model,prep,dict(benchmark_only=True,epochs=0,steps=5,resumable=False,
        source_closure=closure,assets=receipt,execution=execution))
    subset = Prepared([data.encodings[i] for i in batches[-1]],data.numeric[batches[-1]],data.categories[batches[-1]])
    replay_rows = [{k:v for k,v in training[i].items() if k!='y'} for i in batches[-1]]
    delta = replay_gate(out/'checkpoint',model,subset,tokenizer,device,rows=replay_rows)
    checkpoint_seconds = time.perf_counter()-checkpoint_start
    projection = {str(f):project_training(int(((folds>=0)&(folds!=f)).sum()),preparation_seconds,
        load_seconds+initialization_seconds,latencies) for f in range(5)}
    if source_closure()!=closure:
        raise ValueError('source/input changed during benchmark')
    return dict(phase='training_throughput_only',quality_scored=False,confirmation_opened=False,submitted=False,
        train_rows=len(training),train_dialogues=len({r['dialog_id'] for r in training}),
        numeric_columns=prep.numeric_dim,category_columns=14,preparation_seconds=preparation_seconds,
        asset_verification_and_load_seconds=load_seconds,model_initialization_seconds=initialization_seconds,
        step_seconds=latencies,batch_shapes=batch_shapes,
        cold_step_seconds=latencies[0],warm_mean_step_seconds=float(np.mean(latencies[1:])),
        measured_checkpoint_save_and_16_train_row_replay_seconds=checkpoint_seconds,
        checkpoint_max_abs_difference=delta,checkpoint_sha256=file_sha(out/'checkpoint/model.safetensors'),
        checkpoint_metadata_sha256=file_sha(out/'checkpoint/metadata.json'),
        projections_by_fold=projection,projected_full_5fold_warm_seconds=sum(p['warm_seconds'] for p in projection.values()),
        projected_full_5fold_conservative_seconds=sum(p['conservative_seconds'] for p in projection.values()),
        assets=receipt,source_closure=closure,config=CONFIG,device=device,execution=execution)


def pilot_admitted(primary, native):
    return bool(primary<native)


def full_admitted(delta, interval, fold_deltas, short_delta):
    return bool(delta<0 and interval[1]<0 and len(fold_deltas)==5
                and sum(v<0 for v in fold_deltas)>=4 and short_delta<=0)


def require_pilot(path, closure, execution=None):
    report = json.loads((Path(path)/'report.json').read_text())
    if (report.get('phase')!='pilot_fold0' or report.get('admit_other_folds') is not True
            or report.get('source_closure')!=closure or report.get('config')!=CONFIG
            or report.get('quality_authorized') is not True
            or (execution is not None and report.get('execution')!=execution)):
        raise ValueError('passed pilot with unchanged frozen source closure required')
    return report


def require_benchmark(path, closure, execution=None):
    report = json.loads((Path(path)/'report.json').read_text())
    if (report.get('phase')!='training_throughput_only' or report.get('source_closure')!=closure
            or report.get('config')!=CONFIG or report.get('quality_scored') is not False
            or len(report.get('step_seconds',[]))!=5
            or report.get('checkpoint_max_abs_difference',1)>1e-6
            or file_sha(Path(path)/'checkpoint/model.safetensors')!=report.get('checkpoint_sha256')
            or file_sha(Path(path)/'checkpoint/metadata.json')!=report.get('checkpoint_metadata_sha256')
            or (execution is not None and report.get('execution')!=execution)):
        raise ValueError('passed five-batch benchmark with unchanged frozen closure required')
    return report


def verify_controls(rows, folds):
    """Bind historical local checkpoints before any unsafe legacy joblib read."""
    from research.sequence_probe import native_predictions
    native = json.loads((NATIVE/'report.json').read_text())
    guard = json.loads((ROOT/'evidence/guard-correction-v2.json').read_text())
    if native['model']!='template' or native['config']!=experiment.BEHAVIOR_CONFIG:
        raise ValueError('native model/config provenance mismatch')
    if not all(guard['scientific_ast_unchanged'].values()):
        raise ValueError('historical scientific functions changed')
    for name,digest in native['provenance']['sha256'].items():
        if name in guard['old_generation_source']:
            if guard['old_generation_source'][name]!=digest:
                raise ValueError('historical native source mismatch')
            historical = NATIVE/'snapshot'/name
            if file_sha(historical)!=digest:
                raise ValueError('native historical source snapshot mismatch')
        elif name=='BRIEF.md':
            if file_sha(NATIVE/'snapshot'/name)!=digest:
                raise ValueError('historical native BRIEF snapshot mismatch')
        elif file_sha(ROOT/name)!=digest:
            raise ValueError('native historical data/rules/membership mismatch')
    if file_sha(NATIVE/'oof.csv')!=native['oof_sha256']:
        raise ValueError('native OOF binding mismatch')
    if set(native['artifacts'])!={f'fold-{f}.joblib' for f in range(5)}:
        raise ValueError('native checkpoints incomplete')
    for name,digest in native['artifacts'].items():
        if file_sha(NATIVE/name)!=digest:
            raise ValueError('native checkpoint hash mismatch')
    for f,record in enumerate(native['folds']):
        tr,ev = split_indices(folds,f)
        if (record['fold']!=f or record['train_IDs']!=[rows[i]['ID'] for i in tr]
                or record['validation_IDs']!=[rows[i]['ID'] for i in ev]
                or record['train_rows']!=len(tr) or record['validation_rows']!=len(ev)):
            raise ValueError('native exact TRAIN/evaluation membership mismatch')
    old = json.loads((H8/'source-closure.json').read_text())
    for name,digest in old.items():
        if file_sha(ROOT/name)!=digest:
            raise ValueError('H8 historical source/input closure mismatch')
    h8_report = json.loads((H8/'report.json').read_text())
    if h8_report['source_closure']!=old:
        raise ValueError('H8 report source binding mismatch')
    native_map = native_predictions(NATIVE/'oof.csv',rows,folds)
    h8_raw = native_predictions(H8/'sequence.csv',rows,folds)
    h8_primary = native_predictions(H8/'primary.csv',rows,folds)
    for identifier,p in h8_primary.items():
        if abs(p-(.5*native_map[identifier]+.5*h8_raw[identifier]))>1e-12:
            raise ValueError('H8 saved hardsequence mixture differs from exact frozen decoder')
    return native_map,h8_primary


def replay_controls(fold, training, evaluation, prep, native_map, h8_map, device):
    import joblib
    from research import sequence_probe as h8
    model = joblib.load(NATIVE/f'fold-{fold}.joblib')  # Local, hash-bound trusted legacy artifact only.
    unlabelled = [{k:v for k,v in r.items() if k!='y'} for r in evaluation]
    native = experiment.predict('template',model,unlabelled).astype(np.float64)
    expected = np.asarray([native_map[r['ID']] for r in evaluation],dtype=np.float64)
    if not np.array_equal(native,expected):
        raise ValueError('native exact cached checkpoint replay failed')
    membership = json.loads((H8/f'fold-{fold}/membership.json').read_text())
    if (membership['train_IDs']!=[r['ID'] for r in training]
            or membership['evaluation_IDs']!=[r['ID'] for r in evaluation]):
        raise ValueError('H8 training ancestry/membership mismatch')
    historical_device = json.loads((H8/'report.json').read_text())['device']
    if historical_device not in ('cpu','mps'):
        raise ValueError('unrecognized historical control generation device')
    if historical_device=='mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('historical MPS control replay unavailable; no cross-device fallback')
        torch.mps.set_per_process_memory_fraction(CONFIG['mps_memory_fraction'])
    model,old_prep,metadata = h8.load_bundle(H8/f'fold-{fold}/sequence',historical_device)
    if old_prep.to_dict()!=prep.to_dict():
        raise ValueError('H8 full strong bank train-only scaler/vocabulary ancestry mismatch')
    raw = h8.predict(model,unlabelled,old_prep,historical_device).astype(np.float64)
    expected_h8 = np.asarray([h8_map[r['ID']] for r in evaluation],dtype=np.float64)
    delta = float(np.max(np.abs(.5*native+.5*raw-expected_h8)))
    if delta>1e-6:
        raise ValueError('H8 checkpoint replay failed')
    del model
    if historical_device=='mps':
        torch.mps.empty_cache()
    return dict(native_exact_replay=True,h8_checkpoint_max_abs_difference=delta,historical_control_device=historical_device,
                native_train_rows=len(training),h8_train_rows=len(training))


def train_arm(data, prep, tokenizer, device, path, closure, receipt, deadline, data_labels, execution=None):
    execution = execution or dict(micro_batch_size=16,empty_mps_cache=False)
    seed_all()
    encoder,_,verified = load_verified_assets()
    if verified!=receipt:
        raise ValueError('verified assets changed between folds')
    seed_all()
    model = EncoderModel(encoder,prep.numeric_dim,prep.category_sizes).to(device=device,dtype=torch.float32)
    optimizer = optimizer_for(model)
    rng = np.random.default_rng(1729)
    epoch_times = []
    # Targets are supplied separately by quality(), never inherited as network inputs.
    for epoch in range(5):
        begin = time.perf_counter()
        order = rng.permutation(len(data))
        for start in range(0,len(order),16):
            if time.perf_counter()>deadline:
                raise TimeoutError('one-shot measured fit exceeded explicit parent wall timeout; no resume')
            indices = order[start:start+16]
            train_step(model,optimizer,tensor_batch(data,indices,tokenizer,device),
                       torch.as_tensor(data_labels[indices],device=device,dtype=torch.float32),
                       micro_batch_size=execution['micro_batch_size'],empty_mps_cache=execution['empty_mps_cache'])
        synchronize(device)
        epoch_times.append(time.perf_counter()-begin)
        print(json.dumps(dict(epoch=epoch+1,seconds=epoch_times[-1],quality_logged=False)),flush=True)
    del optimizer
    model.zero_grad(set_to_none=True)
    save_bundle(path,model,prep,dict(epoch=5,epoch_seconds=epoch_times,source_closure=closure,
        assets=receipt,resumable=False,benchmark_only=False,execution=execution))
    return model,epoch_times


def quality(out, device, tokenizer, receipt, closure, deadline, full=False, pilot_out=None, execution=None):
    snapshot(out,closure)
    if full:
        require_pilot(pilot_out,closure,execution)
    rows,folds = load_training()
    native_map,h8_map = verify_controls(rows,folds)
    raw_prediction = np.full(len(rows),np.nan,dtype=np.float64)
    checkpoints = []
    for fold in (range(5) if full else [0]):
        itrain,ival = split_indices(folds,fold)
        training,evaluation = [rows[i] for i in itrain],[rows[i] for i in ival]
        if {r['group'] for r in training}&{r['group'] for r in evaluation}:
            raise ValueError('supervised component leakage')
        fold_out = out/f'fold-{fold}'
        fold_out.mkdir()
        membership = dict(train_IDs=[r['ID'] for r in training],evaluation_IDs=[r['ID'] for r in evaluation])
        write_json(fold_out/'membership.json',membership)
        prep,data = fit_preparation(training,tokenizer)
        control_receipt = replay_controls(fold,training,evaluation,prep,native_map,h8_map,device)
        unlabelled = [{k:v for k,v in r.items() if k!='y'} for r in evaluation]
        eval_data = prepare(unlabelled,prep,tokenizer)
        if full and fold==0:
            path = Path(pilot_out)/'fold-0/checkpoint'
            pilot = require_pilot(pilot_out,closure,execution)
            saved = next(r for r in pilot['checkpoints'] if r['fold']==0)
            if (file_sha(path/'model.safetensors')!=saved['checkpoint_sha256']
                    or file_sha(path/'metadata.json')!=saved['metadata_sha256']):
                raise ValueError('passed pilot checkpoint hash binding mismatch')
            old_membership = json.loads((Path(pilot_out)/'fold-0/membership.json').read_text())
            if old_membership!=membership:
                raise ValueError('pilot fold0 exact membership mismatch')
            model,old_prep,metadata = load_bundle(path,device)
            if (metadata['source_closure']!=closure or metadata.get('epoch')!=5
                    or metadata.get('benchmark_only') is not False or old_prep.to_dict()!=prep.to_dict()
                    or metadata.get('execution')!=execution):
                raise ValueError('pilot checkpoint provenance mismatch')
            times = metadata['epoch_seconds']
            reused = True
        else:
            path = fold_out/'checkpoint'
            model,times = train_arm(data,prep,tokenizer,device,path,closure,receipt,deadline,
                                    np.asarray([r['y'] for r in training],dtype=np.float32),execution=execution)
            reused = False
        raw_prediction[ival] = predict_prepared(model,eval_data,tokenizer,device)
        delta = replay_gate(path,model,eval_data,tokenizer,device,rows=unlabelled)
        checkpoints.append(dict(fold=fold,path=str(path.relative_to(ROOT)),epoch_seconds=times,
            checkpoint_max_abs_difference=delta,reused_passed_pilot=reused,
            checkpoint_sha256=file_sha(path/'model.safetensors'),metadata_sha256=file_sha(path/'metadata.json'),
            controls=control_receipt))
        del model
        if device=='mps':
            torch.mps.empty_cache()
    idx = np.flatnonzero(folds>=0) if full else np.flatnonzero(folds==0)
    evaluation = [rows[i] for i in idx]
    components = dict(raw_encoder=raw_prediction[idx],
        native=np.asarray([native_map[r['ID']] for r in evaluation],dtype=np.float64),
        h8_hardsequence=np.asarray([h8_map[r['ID']] for r in evaluation],dtype=np.float64))
    components['primary'] = .5*components['native']+.5*components['raw_encoder']
    from finalize import paired_report
    metrics = {n:experiment.metrics([r['y'] for r in evaluation],np.asarray(p,dtype=np.float64)) for n,p in components.items()}
    for n,p in components.items():
        experiment.write_predictions(out/f'{n}.csv',evaluation,p,folds[idx])
    comparisons = {n:paired_report(evaluation,np.asarray(components[n],dtype=np.float64),components['primary'])
                   for n in ('native','h8_hardsequence')}
    fold_reports = {}
    for fold in sorted(set(folds[idx])):
        mask = folds[idx]==fold
        fold_rows = [r for r,m in zip(evaluation,mask) if m]
        fold_reports[str(fold)] = paired_report(fold_rows,components['native'][mask],components['primary'][mask])
        fold_reports[str(fold)]['versus_h8_hardsequence'] = paired_report(
            fold_rows,components['h8_hardsequence'][mask],components['primary'][mask])
        fold_reports[str(fold)]['metrics'] = {n:experiment.metrics([r['y'] for r in fold_rows],
            np.asarray(p[mask],dtype=np.float64)) for n,p in components.items()}
    labels_by_dialog = {}
    for r in evaluation:
        labels_by_dialog.setdefault(r['dialog_id'],[]).append(r['y'])
    masks = dict(short_2_5=np.asarray([2<=len(r['messages'])<=5 for r in evaluation]),
        long_10_plus=np.asarray([len(r['messages'])>=10 for r in evaluation]),
        human_human=np.asarray([sum(labels_by_dialog[r['dialog_id']])==0 for r in evaluation]))
    masks.update({f'class_{label}':np.asarray([r['y']==label for r in evaluation]) for label in (0,1)})
    slices = {}
    for name,mask in masks.items():
        if mask.any():
            slices[name] = paired_report([r for r,m in zip(evaluation,mask) if m],
                components['native'][mask],components['primary'][mask])
            slices[name]['versus_h8_hardsequence'] = paired_report(
                [r for r,m in zip(evaluation,mask) if m],components['h8_hardsequence'][mask],components['primary'][mask])
    if source_closure()!=closure:
        raise ValueError('frozen source/input closure changed during quality')
    comparison = comparisons['native']
    admitted = pilot_admitted(metrics['primary']['logloss'],metrics['native']['logloss'])
    short_delta = slices.get('short_2_5',{}).get('delta_candidate_minus_base',float('inf'))
    return dict(phase='full_development_5fold' if full else 'pilot_fold0',config=CONFIG,source_closure=closure,
        quality_authorized=True,quality_scored=True,metrics=metrics,comparisons=comparisons,folds=fold_reports,
        slices=slices,checkpoints=checkpoints,admit_other_folds=bool(not full and admitted),
        full_gate=bool(full and full_admitted(comparison['delta_candidate_minus_base'],
            comparison['group_bootstrap_95pct_interval'],[v['delta_candidate_minus_base'] for v in fold_reports.values()],short_delta)),
        confirmation_opened=False,submitted=False,device=device,assets=receipt,execution=execution)


def main(argv=None):
    import argparse
    import platform
    import sys
    import transformers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['benchmark','pilot','full'])
    parser.add_argument('--out',required=True,type=Path,help='new child of runs/encoder-preflight-v2/')
    parser.add_argument('--device',required=True,choices=['mps','cpu'])
    parser.add_argument('--allow-quality',action='store_true',help='parent-only explicit quality admission')
    parser.add_argument('--benchmark-out',type=Path)
    parser.add_argument('--pilot-out',type=Path)
    parser.add_argument('--wall-timeout-seconds',type=float)
    parser.add_argument('--micro-batch-size',type=int,choices=[1,2,4,8,16],default=16)
    parser.add_argument('--empty-mps-cache',action='store_true')
    args = parser.parse_args(argv)
    execution = dict(device=args.device,micro_batch_size=args.micro_batch_size,gradient_accumulation_steps=16//args.micro_batch_size,
                     effective_batch_size=16,empty_mps_cache=args.empty_mps_cache,
                     mps_low_watermark_ratio=os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO'),
                     mps_high_watermark_ratio=os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO'))
    if args.action!='benchmark':
        if not args.allow_quality or args.benchmark_out is None or not args.wall_timeout_seconds or args.wall_timeout_seconds<=0:
            parser.error('quality requires --allow-quality --benchmark-out and explicit positive --wall-timeout-seconds')
        if args.action=='full' and args.pilot_out is None:
            parser.error('full requires a passed --pilot-out')
    elif args.allow_quality:
        parser.error('benchmark must not enable quality')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='4':
            parser.error(f'launch with {name}=4; CPU resource configuration must be explicit before imports')
    if os.environ.get('TMPDIR')!='/Users/Ivan/.hermes/profiles/hackatons/cache/scratch':
        parser.error('explicit approved TMPDIR required')
    configure(args.device)
    started = time.perf_counter()
    out = new_output(args.out)
    resources = ResourceSampler(args.device)
    try:
        with no_network(),resources:
            begin = time.perf_counter()
            encoder,tokenizer,receipt = load_verified_assets()
            load_seconds = time.perf_counter()-begin
            closure = source_closure()
            if args.action=='benchmark':
                report = benchmark(out,args.device,encoder,tokenizer,receipt,closure,load_seconds,execution)
            else:
                # Never keep a spare pretrained model alongside the actual fit.
                del encoder
                require_benchmark(args.benchmark_out,closure,execution)
                if args.action=='full':
                    require_pilot(args.pilot_out,closure,execution)
                report = quality(out,args.device,tokenizer,receipt,closure,
                    started+args.wall_timeout_seconds,args.action=='full',args.pilot_out,execution=execution)
                report['benchmark_report_sha256'] = file_sha(args.benchmark_out/'report.json')
                report['parent_wall_timeout_seconds'] = args.wall_timeout_seconds
                if args.pilot_out is not None:
                    report['pilot_report_sha256'] = file_sha(args.pilot_out/'report.json')
        report['resources'] = resources.report()
        report['wall_seconds'] = time.perf_counter()-started
        report['runtime'] = dict(python=platform.python_version(),executable=sys.executable,
            torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
            cpu_threads=torch.get_num_threads(),mps_memory_fraction=(CONFIG['mps_memory_fraction'] if args.device=='mps' else None),device=args.device,
            dtype='float32',autocast=False,attention_implementation='eager',
            network_disabled=True,local_files_only=True,trust_remote_code=False,use_safetensors=True,
            environment={n:os.environ.get(n) for n in ('TMPDIR','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
                'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','HF_HUB_OFFLINE',
                'TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM')})
        report['config_sha256'] = file_sha(out/'config.json')
        write_json(out/'report.json',report)
        summary = {k:v for k,v in report.items() if k not in ('source_closure','resources','assets')}
        summary['sampled_peaks'] = report['resources']['sampled_peaks']
        print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
        return report
    except Exception as exc:
        write_json(out/'failure.json',dict(action=args.action,exception_type=type(exc).__name__,
            error=str(exc),quality_scored=False if args.action=='benchmark' else 'unknown',
            wall_seconds=time.perf_counter()-started,resources=resources.report(),
            resumable=False,submitted=False))
        raise


if __name__=='__main__':
    main()
