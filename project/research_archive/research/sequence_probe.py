"""Frozen H8 v2: local-only sequence probe; dialogues are untrusted data."""
from __future__ import annotations

import numpy as np


def encode_messages(messages):
    ordered = sorted(messages, key=lambda m: m['message'])
    if len(ordered) > 16:
        ordered = ordered[:8] + ordered[-8:]
    tokens = np.zeros((16, 256), dtype=np.int16)
    roles = np.zeros(16, dtype=np.int64)
    mask = np.zeros(16, dtype=bool)
    for i, message in enumerate(ordered):
        if message['participant_index'] not in (0, 1):
            raise ValueError('invalid actual participant role')
        import re
        text = re.sub(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b',
                      ' UUIDVALUE ', message['text'])
        raw = text.encode('utf-8')
        if len(raw) > 256:
            raw = raw[:192] + raw[-64:]
        tokens[i, :len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) + 1
        roles[i] = message['participant_index'] + 1
        mask[i] = True
    return tokens, roles, mask


from collections import Counter
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import experiment


class Preprocessor:
    def fit_bank(self, bank):
        raw = np.asarray(bank[:, :-14], dtype=np.float64)
        logged = np.sign(raw) * np.log1p(np.abs(raw))
        if not np.isfinite(logged).all():
            raise ValueError('non-finite feature bank')
        self.scaler = StandardScaler().fit(logged)
        self.vocabs = []
        for column in bank[:, -14:].T:
            counts = Counter(map(str, column))
            self.vocabs.append({v: i + 1 for i, v in enumerate(sorted(v for v, n in counts.items() if n >= 2))})
        return self

    @property
    def numeric_dim(self):
        return len(self.scaler.mean_)

    @property
    def category_sizes(self):
        return [len(v) + 1 for v in self.vocabs]

    def transform_bank(self, bank):
        if bank.shape[1] != self.numeric_dim + 14:
            raise ValueError('template feature schema changed')
        raw = np.asarray(bank[:, :-14], dtype=np.float64)
        logged = np.sign(raw) * np.log1p(np.abs(raw))
        numeric = np.clip(self.scaler.transform(logged), -8, 8).astype(np.float32)
        categories = np.asarray([[v.get(str(value), 0) for v, value in zip(self.vocabs, row)]
                                 for row in bank[:, -14:]], dtype=np.int64)
        if not np.isfinite(numeric).all():
            raise ValueError('non-finite transformed features')
        return numeric, categories

    def to_dict(self):
        return dict(mean=self.scaler.mean_.tolist(), scale=self.scaler.scale_.tolist(), vocabs=self.vocabs)

    @classmethod
    def from_dict(cls, metadata):
        obj = cls()
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = np.asarray(metadata['mean'])
        obj.scaler.scale_ = np.asarray(metadata['scale'])
        obj.scaler.n_features_in_ = len(obj.scaler.mean_)
        obj.vocabs = metadata['vocabs']
        return obj


@dataclass
class Prepared:
    inputs: dict
    row_slots: np.ndarray
    rows: list
    dialog_ids: list

    def labels(self):
        result = np.zeros((len(self.dialog_ids), 2), dtype=np.float32)
        for row, (d, p) in zip(self.rows, self.row_slots):
            result[d, p] = row['y']
        return result


def prepare(rows, prep, bank=None):
    numeric, categories = prep.transform_bank(experiment.model_input(rows, 'template') if bank is None else bank)
    dialog_ids = list(dict.fromkeys(r['dialog_id'] for r in rows))
    index = {d: i for i, d in enumerate(dialog_ids)}
    slots = np.asarray([(index[r['dialog_id']], r['participant_index']) for r in rows], dtype=np.int64)
    if len(rows) != 2 * len(dialog_ids) or len(set(map(tuple, slots))) != len(rows):
        raise ValueError('require exactly two distinct participant rows per dialogue')
    n = len(dialog_ids)
    arrays = dict(tokens=np.zeros((n, 16, 256), dtype=np.int16), roles=np.zeros((n, 16), dtype=np.int64),
                  mask=np.zeros((n, 16), dtype=bool), numeric=np.zeros((n, 2, prep.numeric_dim), dtype=np.float32),
                  categories=np.zeros((n, 2, 14), dtype=np.int64))
    seen = {}
    for i, (row, (d, p)) in enumerate(zip(rows, slots)):
        arrays['numeric'][d, p], arrays['categories'][d, p] = numeric[i], categories[i]
        if d not in seen:
            arrays['tokens'][d], arrays['roles'][d], arrays['mask'][d] = encode_messages(row['messages'])
            seen[d] = row['messages']
        elif row['messages'] != seen[d]:
            raise ValueError('dialogue participant message mismatch')
    return Prepared(arrays, slots, rows, dialog_ids)


import json
import random
from pathlib import Path
import torch
from torch import nn
from botbench import file_sha

CONFIG = dict(version='H8-sequence-v2', seed=1729, epochs=25, batch_size=32,
              learning_rate=.001, weight_decay=.01, gradnorm=1., cpu_threads=4,
              mps_memory_fraction=.35, byte_embedding=16, role_embedding=8,
              kernels=[3, 5, 7], channels=32, hidden=96, head=128, dropout=.2,
              max_messages=16, max_bytes=256, min_category_count=2,
              primary_weight=.5, checkpoint_tolerance=1e-6)


def seed_all():
    random.seed(1729)
    np.random.seed(1729)
    torch.manual_seed(1729)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1729)


def configure(device):
    torch.set_num_threads(4)
    if device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable; no silent fallback')
        torch.mps.set_per_process_memory_fraction(.35)
    elif device != 'cpu':
        raise ValueError('device must be explicit cpu or mps')
    seed_all()
    return device


class SequenceModel(nn.Module):
    def __init__(self, numeric_dim, category_sizes, mode='sequence'):
        super().__init__()
        if mode not in ('sequence', 'no_sequence'):
            raise ValueError('invalid ablation')
        self.mode, self.numeric_dim, self.category_sizes = mode, numeric_dim, list(category_sizes)
        self.byte_embedding = nn.Embedding(257, 16, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(16, 32, k, padding=k // 2) for k in (3, 5, 7)])
        self.role_embedding = nn.Embedding(3, 8, padding_idx=0)
        self.gru = nn.GRU(104, 96, batch_first=True)
        self.category_embeddings = nn.ModuleList([nn.Embedding(n, 8) for n in category_sizes])
        self.head = nn.Sequential(nn.Linear(288 + numeric_dim + 14 * 8, 128), nn.GELU(),
                                  nn.Dropout(.2), nn.Linear(128, 1))

    def forward(self, tokens, roles, mask, numeric, categories):
        b, length, width = tokens.shape
        sequence = numeric.new_zeros((b, 2, 288))
        if self.mode == 'sequence':
            embedded = self.byte_embedding(tokens.reshape(b * length, width)).transpose(1, 2)
            byte_mask = (tokens.reshape(b * length, width) != 0).unsqueeze(1)
            pooled = []
            for conv in self.convs:
                h = torch.nn.functional.gelu(conv(embedded))
                h = h.masked_fill(~byte_mask, -1e4).amax(dim=2)
                h = h * byte_mask.any(dim=2).to(h.dtype)
                pooled.append(h)
            messages = torch.cat(pooled, dim=1).reshape(b, length, 96)
            encoded, _ = self.gru(torch.cat([messages, self.role_embedding(roles)], dim=2))
            last_idx = mask.long().sum(dim=1).clamp(min=1) - 1
            last = encoded[torch.arange(b, device=encoded.device), last_idx]
            means = []
            for p in (0, 1):
                weights = (mask & (roles == p + 1)).unsqueeze(-1).to(encoded.dtype)
                means.append((encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1))
            sequence = torch.stack([torch.cat([means[p], means[1-p], last], dim=1) for p in (0, 1)], dim=1)
        cats = torch.cat([emb(categories[:, :, i]) for i, emb in enumerate(self.category_embeddings)], dim=2)
        return self.head(torch.cat([sequence, numeric, cats], dim=2)).squeeze(-1)


def tensor_batch(data, indices, device):
    return {key: torch.as_tensor(value[indices], dtype=(torch.bool if key == 'mask' else
            torch.float32 if key == 'numeric' else torch.long), device=device)
            for key, value in data.inputs.items()}


def train_step(model, optimizer, batch, labels):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.binary_cross_entropy_with_logits(model(**batch), labels)
    if not torch.isfinite(loss):
        raise ValueError('non-finite training loss')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return loss.detach()


def predict_prepared(model, data, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(data.dialog_ids), 32):
            idx = np.arange(start, min(start + 32, len(data.dialog_ids)))
            parts.append(torch.sigmoid(model(**tensor_batch(data, idx, device))).cpu().numpy())
    canonical = np.concatenate(parts, axis=0)
    return canonical[data.row_slots[:, 0], data.row_slots[:, 1]]


def predict(model, rows, prep, device):
    return predict_prepared(model, prepare(rows, prep), device)


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def save_bundle(path, model, prep, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    temporary = path / 'weights.pt.partial'
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
    temporary.rename(path / 'weights.pt')
    write_json(path / 'metadata.json', dict(config=CONFIG, mode=model.mode,
        numeric_dim=model.numeric_dim, category_sizes=model.category_sizes,
        preprocessing=prep.to_dict(), weights_sha256=file_sha(path / 'weights.pt'), metadata=metadata))


def load_bundle(path, device):
    path = Path(path)
    info = json.loads((path / 'metadata.json').read_text())
    if info['config'] != CONFIG or file_sha(path / 'weights.pt') != info['weights_sha256']:
        raise ValueError('checkpoint config/hash binding mismatch')
    prep = Preprocessor.from_dict(info['preprocessing'])
    model = SequenceModel(info['numeric_dim'], info['category_sizes'], info['mode'])
    model.load_state_dict(torch.load(path / 'weights.pt', map_location='cpu', weights_only=True))
    return model.to(device), prep, info['metadata']


import argparse
import hashlib
import os
import platform
import shutil
import threading
import time
import psutil
from botbench import load_dataset, membership_folds, read_probabilities

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'runs/template-v1/oof.csv'


def new_output(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def split_indices(folds, fold):
    folds = np.asarray(folds)
    return np.flatnonzero((folds >= 0) & (folds != fold)), np.flatnonzero(folds == fold)


def pilot_admitted(primary, native, control):
    return bool(primary < native and primary < control)


def warm_projection(dialogues, preparation_seconds, results):
    steps = int(np.ceil(dialogues / 32)) * 25
    return float(preparation_seconds + sum(r['step_seconds'][0] +
        (steps - 1) * np.mean(r['step_seconds'][1:]) for r in results.values()))


def native_predictions(path, rows, folds):
    import csv
    values = read_probabilities(path, has_fold=True)
    expected = {r['ID']: int(f) for r, f in zip(rows, folds) if f >= 0}
    with Path(path).open(newline='', encoding='utf-8') as f:
        actual = {r['ID']: int(r['fold']) for r in csv.DictReader(f)}
    if actual != expected:
        raise ValueError('native OOF membership/fold mismatch')
    return values


def source_closure():
    names = ['botbench.py', 'experiment.py', 'llm_bridge.py', 'finalize.py', 'campaign.py',
             'research/sequence_probe.py', 'tests/test_sequence_probe.py', 'requirements.txt',
             'evidence/membership-v2.json', 'evidence/rules.txt', 'evidence/sequence-probe-v2-prereg.md',
             'data/train.json', 'data/ytrain.csv', 'runs/template-v1/oof.csv', 'runs/template-v1/report.json']
    return {name: file_sha(ROOT / name) for name in names}


def snapshot(out, closure):
    for name, digest in closure.items():
        target = out / 'snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_sha(target) != digest:
            raise ValueError('source changed during snapshot')
    write_json(out / 'config.json', CONFIG)
    write_json(out / 'source-closure.json', closure)


class Resources:
    def __init__(self, device):
        self.device, self.samples = device, []
        self.stop_event = threading.Event()
        self.started = time.perf_counter()

    def sample(self):
        row = dict(seconds=time.perf_counter() - self.started, rss_bytes=psutil.Process().memory_info().rss)
        if self.device == 'mps':
            row.update(mps_current_bytes=torch.mps.current_allocated_memory(),
                       mps_driver_bytes=torch.mps.driver_allocated_memory())
        self.samples.append(row)

    def loop(self):
        while not self.stop_event.wait(.1):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def report(self):
        keys = ['rss_bytes'] + (['mps_current_bytes', 'mps_driver_bytes'] if self.device == 'mps' else [])
        return dict(sample_interval_seconds=.1, samples=self.samples,
                    sampled_peaks={key: max(s[key] for s in self.samples) for key in keys})


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def load_training():
    rows = load_dataset(ROOT / 'data', 'train')
    membership = json.loads((ROOT / 'evidence/membership-v2.json').read_text())
    folds = np.asarray(membership_folds(rows, membership))
    return rows, folds


def fit_preparation(rows):
    bank = experiment.model_input(rows, 'template')
    prep = Preprocessor().fit_bank(bank)
    return prep, prepare(rows, prep, bank)


def replay_gate(path, model, prep, rows, device):
    before = predict(model, rows, prep, device)
    loaded, loaded_prep, _ = load_bundle(path, device)
    after = predict(loaded, rows, loaded_prep, device)
    delta = float(np.max(np.abs(before - after)))
    if delta > 1e-6:
        raise ValueError(f'checkpoint replay failed: {delta}')
    return delta


def benchmark(out, device):
    started = time.perf_counter()
    closure = source_closure()
    snapshot(out, closure)
    rows, folds = load_training()
    selected = np.flatnonzero(folds > 0)
    training = [rows[i] for i in selected]
    prep, data = fit_preparation(training)
    preparation_seconds = time.perf_counter() - started
    # Unsupervised length-stratified cases plus a longest real dialogue.
    lengths = {r['dialog_id']: sum(len(m['text'].encode('utf-8')) for m in r['messages']) for r in training}
    ranked = np.argsort([lengths[d] for d in data.dialog_ids], kind='stable')
    if len(ranked) < 160:
        raise ValueError('benchmark needs five complete disjoint 32-dialogue batches')
    chunks = np.array_split(ranked, 5)
    rng = np.random.default_rng(1729)
    batches = [rng.choice(chunk, 32, replace=False) for chunk in chunks]
    longest = int(ranked[-1])
    if longest not in batches[-1]:
        batches[-1][-1] = longest
    write_json(out / 'membership.json', dict(train_IDs=[r['ID'] for r in training], evaluation_IDs=[],
        excluded_folds=[-1, 0], batches=[[data.dialog_ids[i] for i in b] for b in batches],
        longest_dialog_id=data.dialog_ids[longest], longest_raw_bytes=int(lengths[data.dialog_ids[longest]])))
    results = {}
    labels = data.labels()
    for mode in ('sequence', 'no_sequence'):
        seed_all()
        model = SequenceModel(prep.numeric_dim, prep.category_sizes, mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        latencies = []
        for batch_idx, idx in enumerate(batches):
            synchronize(device)
            begin = time.perf_counter()
            batch = tensor_batch(data, idx, device)
            target = torch.as_tensor(labels[idx], device=device)
            train_step(model, optimizer, batch, target)
            synchronize(device)
            elapsed = time.perf_counter() - begin
            latencies.append(elapsed)
            print(json.dumps(dict(mode=mode, train_batch=batch_idx + 1, seconds=elapsed, dialogues=len(idx))), flush=True)
        path = out / mode
        save_bundle(path, model, prep, dict(epoch=0, steps=5, benchmark_only=True, source_closure=closure, resumable=False))
        sample_ids = {data.dialog_ids[i] for i in batches[-1]}
        sample_rows = [{k:v for k,v in r.items() if k != 'y'} for r in training if r['dialog_id'] in sample_ids]
        delta = replay_gate(path, model, prep, sample_rows, device)
        results[mode] = dict(step_seconds=latencies, mean_step_seconds=float(np.mean(latencies)),
            warm_mean_step_seconds=float(np.mean(latencies[1:])), max_step_seconds=max(latencies),
            checkpoint_max_abs_difference=delta, checkpoint_sha256=file_sha(path / 'weights.pt'))
        del model, optimizer
        if device == 'mps':
            torch.mps.empty_cache()
    steps = int(np.ceil(len(data.dialog_ids) / 32)) * 25
    projection = preparation_seconds + steps * sum(r['mean_step_seconds'] for r in results.values())
    upper = preparation_seconds + steps * sum(r['max_step_seconds'] for r in results.values())
    if source_closure() != closure:
        raise ValueError('source/input bytes changed during benchmark')
    return dict(phase='training_throughput_only', device=device, config=CONFIG, source_closure=closure,
        preparation_seconds=preparation_seconds, train_rows=len(training), train_dialogues=len(data.dialog_ids),
        numeric_columns=prep.numeric_dim, category_columns=14, results=results,
        projected_pilot_warm_training_seconds=warm_projection(len(data.dialog_ids), preparation_seconds, results),
        projected_pilot_25epoch_two_arm_training_seconds=projection,
        projected_using_max_step_seconds=upper,
        projection_excludes='evaluation preparation/inference, checkpoint IO and native bootstrap reports',
        seconds=time.perf_counter()-started, quality_scored=False, confirmation_opened=False, submitted=False)


def train_arm(data, prep, mode, path, device, closure):
    seed_all()
    model = SequenceModel(prep.numeric_dim, prep.category_sizes, mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    labels, rng = data.labels(), np.random.default_rng(1729)
    epoch_times = []
    for epoch in range(25):
        begin = time.perf_counter()
        order = rng.permutation(len(data.dialog_ids))
        for start in range(0, len(order), 32):
            idx = order[start:start + 32]
            train_step(model, optimizer, tensor_batch(data, idx, device), torch.as_tensor(labels[idx], device=device))
        synchronize(device)
        epoch_times.append(time.perf_counter()-begin)
        print(json.dumps(dict(mode=mode, epoch=epoch+1, seconds=epoch_times[-1])), flush=True)
    save_bundle(path, model, prep, dict(epoch=25, epoch_seconds=epoch_times, source_closure=closure, resumable=False))
    return model, epoch_times


def quality(out, device, full=False, pilot_out=None):
    # Only parent may dispatch this path after reviewing frozen code/benchmark.
    closure = source_closure()
    if full:
        if pilot_out is None:
            raise ValueError('full requires an admitted --pilot-out')
        pilot = json.loads((Path(pilot_out) / 'report.json').read_text())
        if pilot['phase'] != 'pilot_fold0' or not pilot['admit_other_folds'] or pilot['source_closure'] != closure:
            raise ValueError('pilot not admitted or source closure changed')
    snapshot(out, closure)
    rows, folds = load_training()
    native_map = native_predictions(NATIVE, rows, folds)
    predictions = {mode: np.full(len(rows), np.nan) for mode in ('sequence', 'no_sequence')}
    receipts = []
    for fold in (range(5) if full else [0]):
        itrain, ival = split_indices(folds, fold)
        training, evaluation = [rows[i] for i in itrain], [rows[i] for i in ival]
        if {r['group'] for r in training} & {r['group'] for r in evaluation}:
            raise ValueError('component leakage')
        fold_out = out / f'fold-{fold}'
        fold_out.mkdir()
        write_json(fold_out / 'membership.json', dict(train_IDs=[r['ID'] for r in training],
                   evaluation_IDs=[r['ID'] for r in evaluation]))
        prep, data = fit_preparation(training)
        unlabelled = [{k:v for k,v in r.items() if k != 'y'} for r in evaluation]
        eval_data = prepare(unlabelled, prep)
        for mode in predictions:
            path = fold_out / mode
            model, times = train_arm(data, prep, mode, path, device, closure)
            predictions[mode][ival] = predict_prepared(model, eval_data, device)
            delta = replay_gate(path, model, prep, unlabelled, device)
            receipts.append(dict(fold=fold, mode=mode, epoch_seconds=times, checkpoint_max_abs_difference=delta))
            del model
            if device == 'mps':
                torch.mps.empty_cache()
    idx = np.flatnonzero(folds >= 0) if full else np.flatnonzero(folds == 0)
    evaluation = [rows[i] for i in idx]
    native = np.asarray([native_map[r['ID']] for r in evaluation])
    components = dict(native=native, sequence=predictions['sequence'][idx], no_sequence=predictions['no_sequence'][idx])
    components['primary'] = .5 * native + .5 * components['sequence']
    components['control'] = .5 * native + .5 * components['no_sequence']
    from finalize import paired_report
    metrics = {name: experiment.metrics([r['y'] for r in evaluation], p) for name,p in components.items()}
    for name,p in components.items():
        experiment.write_predictions(out / f'{name}.csv', evaluation, p, folds[idx])
    comparisons = dict(primary_vs_native=paired_report(evaluation, native, components['primary']),
                       primary_vs_control=paired_report(evaluation, components['control'], components['primary']))
    fold_comparisons = {}
    for f in sorted(set(folds[idx])):
        mask = folds[idx] == f
        erows = [r for r,m in zip(evaluation,mask) if m]
        fold_comparisons[str(f)] = paired_report(erows, native[mask], components['primary'][mask])
    slices = {}
    masks = {'short_2_5': [2 <= len(r['messages']) <= 5 for r in evaluation],
             'long_10_plus': [len(r['messages']) >= 10 for r in evaluation]}
    dialog_labels = {r['dialog_id']: [] for r in evaluation}
    for r in evaluation:
        dialog_labels[r['dialog_id']].append(r['y'])
    masks['human_human'] = [sum(dialog_labels[r['dialog_id']]) == 0 for r in evaluation]
    for label in (0, 1):
        masks[f'class_{label}'] = [r['y'] == label for r in evaluation]
    for name, values in masks.items():
        mask = np.asarray(values)
        if mask.any():
            slices[name] = paired_report([r for r,m in zip(evaluation,mask) if m], native[mask], components['primary'][mask])
    if source_closure() != closure:
        raise ValueError('source/input changed during quality run')
    admitted = pilot_admitted(metrics['primary']['logloss'], metrics['native']['logloss'], metrics['control']['logloss'])
    return dict(phase='full_development_5fold' if full else 'pilot_fold0', source_closure=closure, config=CONFIG,
        metrics=metrics, comparisons=comparisons, folds=fold_comparisons, slices=slices, checkpoints=receipts,
        admit_other_folds=admitted if not full else False,
        full_gate=bool(full and admitted and comparisons['primary_vs_native']['group_bootstrap_95pct_interval'][1] < 0
                       and sum(v['delta_candidate_minus_base'] < 0 for v in fold_comparisons.values()) >= 4),
        confirmation_opened=False, submitted=False, device=device)


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['benchmark', 'pilot', 'full'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--pilot-out', type=Path)
    args = parser.parse_args(argv)
    if args.device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('disable PYTORCH_ENABLE_MPS_FALLBACK; no silent CPU fallback allowed')
    configure(args.device)
    out = new_output(args.out)
    with Resources(args.device) as resources:
        if args.action == 'benchmark':
            report = benchmark(out, args.device)
        else:
            report = quality(out, args.device, args.action == 'full', args.pilot_out)
    report['resources'] = resources.report()
    report['wall_seconds'] = time.perf_counter() - started
    report['runtime'] = dict(python=platform.python_version(), torch=torch.__version__,
                           numpy=np.__version__, cpu_threads=torch.get_num_threads())
    report['config_sha256'] = file_sha(out / 'config.json')
    write_json(out / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_closure', 'resources')}, indent=2), flush=True)
    print(json.dumps(dict(sampled_peaks=report['resources']['sampled_peaks'])), flush=True)
    return report


if __name__ == '__main__':
    main()
