"""Only synthetic component arithmetic/receipts; no official training or model calls."""
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def runner():
    assert (ROOT / 'run_ensemble.py').is_file(), 'one-command ensemble runner missing'
    return importlib.import_module('run_ensemble')


def test_fixed_half_blend_preserves_sample_order_utf8_crlf_and_no_overwrite(tmp_path):
    module = runner()
    native = {'b': 0.2, 'a': 0.4}
    remote = {'a': 0.8, 'b': 0.6}
    path = tmp_path / 'submission.csv'
    result = module.write_ensemble_csv(path, ['a', 'b'], native, remote)
    assert path.read_bytes() == b'ID,is_bot\r\na,0.6000000000000001\r\nb,0.4\r\n'
    assert result['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        module.write_ensemble_csv(path, ['a', 'b'], native, remote)


@pytest.mark.parametrize('case', ['bool', 'nan', 'membership', 'duplicate', 'historical_hash'])
def test_invalid_final_payload_never_creates_csv(tmp_path, case):
    module = runner()
    native, remote, ids = {'a': 0.2}, {'a': 0.6}, ['a']
    if case == 'bool': native['a'] = True
    if case == 'nan': remote['a'] = float('nan')
    if case == 'membership': remote['b'] = 0.5
    if case == 'duplicate': ids *= 2
    path = tmp_path / 'not-created.csv'
    with pytest.raises(ValueError):
        module.write_ensemble_csv(path, ids, native, remote, expected_sha='0'*64 if case == 'historical_hash' else None)
    assert not path.exists()


from test_adapter import artificial_rows


@pytest.fixture
def synthetic_chain(tmp_path, artificial_rows, monkeypatch):
    """A mock native producer, explicitly NOT six real fits or competition metrics."""
    module = runner()
    from tracking import RunTracker
    from llm import adapter
    train, test = artificial_rows
    ids = [r['ID'] for r in reversed(test)]
    priors = {i: 0.25 for i in ids}
    initial = adapter.prepare_batches(train, test, priors, tmp_path / 'reference-payloads')
    manifest = {'source_batch_digests': {b.name: {k: json.loads((b/'prepared.json').read_text())[k]
                  for k in ('input_sha256', 'prompt_sha256')} for b in initial}}
    components = {i: dict(catboost=0.25, llm=0.6) for i in ids}
    stream = io.StringIO(newline=''); writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow(['ID','is_bot']); writer.writerows((i, 0.5*0.25+0.5*0.6) for i in ids)
    config = json.loads((ROOT/'llm/config.json').read_text())
    config['submission_sha256'] = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    calls = []
    def mock_native(data, output, run_id):
        calls.append('mock_native_no_fit')
        with RunTracker(output, run_id) as child:
            child.params('environment', dict(synthetic_fixture=True))
            for name in ['fold-0', 'fold-1', 'fold-2', 'fold-3', 'fold-4', 'full']:
                child.params('model/'+name, dict(get_params={'iterations':1000,'random_seed':1729},
                             get_all_params={'iterations':1000,'random_seed':1729},tree_count=1000))
                p=child.path('models/'+name+'.cbm'); p.parent.mkdir(exist_ok=True)
                p.write_bytes(b'SYNTHETIC FIXTURE, NOT A TRAINED MODEL')
            child.metric('oof/logloss', 0.25)
            with child.path('native-test.csv').open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['ID','is_bot']); w.writerows(priors.items())
            child.write_json('report.json',dict(synthetic_fixture=True,checkpoint_replay={'all_exact':True},
                             metrics={'logloss':0.25},counts={'train_rows':len(train),'test_rows':len(test)}))
            child.finish()
    monkeypatch.setattr(module, 'native_run', mock_native, raising=False)
    return module, train, test, ids, manifest, components, config, calls


def test_cached_chain_retrains_native_binds_payloads_and_tracks_real_sql_json(tmp_path, synthetic_chain):
    module, train, test, ids, manifest, components, config, calls = synthetic_chain
    assert hasattr(module, 'execute_chain'), 'tracked ensemble chain missing'
    from tracking import RunTracker
    with RunTracker(tmp_path/'outputs', 'synthetic') as tracking:
        report = module.execute_chain(tracking, tmp_path/'data', train, test, ids, 'cached', config,
                                      manifest, components, lambda: None)
        tracking.finish()
    receipt=json.loads((tracking.directory/'tracking.json').read_text())
    with sqlite3.connect(tracking.directory/'tracking.sqlite3') as sql:
        assert sql.execute('select status from runs').fetchone()[0] == 'FINISHED'
    assert calls == ['mock_native_no_fit']
    assert receipt['status']=='FINISHED' and report['llm_mode']=='cached'
    assert receipt['metrics']=={'catboost_oof/logloss':0.25}
    assert len([n for n in receipt['params'] if n.startswith('catboost/model/')])==6
    assert report['ensemble_cv_evaluated'] is False and report['test_labels_read'] is False
    assert (tracking.directory/'submission.csv').is_file() and (tracking.directory/'index.html').is_file()
    assert 'llm/batch-00/prompt.txt' in receipt['artifacts']
    page=(tracking.directory/'index.html').read_text()
    assert 'catboost_oof/logloss' in page and 'llm/batch-00/prompt.txt' in page


@pytest.mark.parametrize('case', ['prior', 'prompt', 'inputs'])
def test_failed_binding_preserves_failed_receipt_without_submission(tmp_path, synthetic_chain, case):
    module, train, test, ids, manifest, components, config, calls = synthetic_chain
    assert hasattr(module, 'execute_chain'), 'tracked ensemble chain missing'
    from tracking import RunTracker
    if case=='prior': components[ids[0]]['catboost'] += 1e-15
    if case=='prompt': manifest['source_batch_digests']['batch-00']['prompt_sha256']='0'*64
    def check():
        if case=='inputs': raise ValueError('raw inputs changed')
    with pytest.raises(ValueError):
        with RunTracker(tmp_path/'outputs', 'failed') as tracking:
            module.execute_chain(tracking,tmp_path/'data',train,test,ids,'cached',config,manifest,components,check)
    receipt=json.loads((tracking.directory/'tracking.json').read_text())
    assert receipt['status']=='FAILED' and not (tracking.directory/'submission.csv').exists()
    with pytest.raises(FileExistsError): RunTracker(tmp_path/'outputs','failed')


def test_cli_requires_explicit_mode_and_fresh_preflight_precedes_native(tmp_path, monkeypatch):
    module=runner()
    assert hasattr(module,'run') and (ROOT/'run_ensemble.sh').is_file(), 'one-command entry missing'
    result=subprocess.run(['bash',str(ROOT/'run_ensemble.sh'),'--data-dir',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==2 and '--llm-mode' in result.stderr
    monkeypatch.setattr(module.adapter,'codex_preflight',lambda: (_ for _ in ()).throw(ValueError('no approved account')))
    monkeypatch.setattr(module,'native_run',lambda *a,**k: pytest.fail('must not train'))
    with pytest.raises(ValueError): module.run(tmp_path/'data',tmp_path/'outputs','preflight','fresh')
    receipt=json.loads((tmp_path/'outputs/preflight/tracking.json').read_text())
    assert receipt['status']=='FAILED' and receipt['error']['stage']=='preflight_codex'


def test_fresh_composition_uses_new_outputs_without_historical_hash_gate(tmp_path,synthetic_chain,monkeypatch):
    module,train,test,ids,manifest,components,config,calls=synthetic_chain
    from tracking import RunTracker
    tool=tmp_path/'codex'; tool.write_text('synthetic executable')
    cli=dict(executable=str(tool),executable_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),version='0.163.0')
    external_calls=[]
    def external(command,**kwargs):
        external_calls.append(command)
        batch=Path(command[command.index('--output-last-message')+1]).parent
        dids=json.loads((batch/'prepared.json').read_text())['dialog_ids']
        (batch/'output.json').write_text(json.dumps({'predictions':[dict(dialog_id=d,p0=0.6,p1=0.4,reason='synthetic') for d in dids]}))
        kwargs['stdout'].write(json.dumps({'item':{'type':'agent_message'}})+'\n')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(module.adapter.subprocess,'run',external)
    with RunTracker(tmp_path/'outputs','fresh-synthetic') as tracking:
        report=module.execute_chain(tracking,tmp_path/'data',train,test,ids,'fresh',config,manifest,components,lambda:None,cli)
        tracking.finish()
    receipt=json.loads((tracking.directory/'tracking.json').read_text())
    assert receipt['status']=='FINISHED' and len(external_calls)==1
    assert report['historical_hash_is_success_gate'] is False and report['historical_csv_matches'] is False
    assert report['historical_public_logloss'] is None and report['fresh_official_score'] is None
    assert calls==['mock_native_no_fit'] and receipt['params']['llm_origin']['new_llm_calls']==1


@pytest.mark.parametrize('case',['checkpoint','sql_json','status','order'])
def test_native_readback_or_order_failure_blocks_llm(tmp_path,synthetic_chain,monkeypatch,case):
    module,train,test,ids,manifest,components,config,calls=synthetic_chain
    from tracking import RunTracker
    original=module.native_run
    def bad_native(data,output,name):
        original(data,output,name)
        target=output/name
        if case=='checkpoint': (target/'models/full.cbm').write_bytes(b'tampered')
        elif case in {'sql_json','status'}:
            receipt=json.loads((target/'tracking.json').read_text())
            if case=='status': receipt['status']='FAILED'
            else: receipt['params']['model/full']['tree_count']=9
            (target/'tracking.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(module,'native_run',bad_native)
    monkeypatch.setattr(module.adapter,'prepare_batches',lambda *a,**k:pytest.fail('must block before LLM preparation'))
    if case=='order': ids=list(reversed(ids))
    with pytest.raises(ValueError):
        with RunTracker(tmp_path/'outputs','failed-native') as tracking:
            module.execute_chain(tracking,tmp_path/'data',train,test,ids,'cached',config,manifest,components,lambda:None)
    assert json.loads((tracking.directory/'tracking.json').read_text())['status']=='FAILED'
    assert not (tracking.directory/'submission.csv').exists()


def test_output_paths_cannot_put_raw_payloads_in_coursework_export_or_data(tmp_path):
    module=runner()
    assert hasattr(module,'validate_output_location'), 'raw payload output confinement missing'
    for output in [ROOT/'evidence/new-raw-payloads',tmp_path/'official-data/runs']:
        with pytest.raises(ValueError,match='output'):
            module.run(tmp_path/'official-data',output,'must-not-create','cached')
        assert not output.exists()


def test_public_fresh_wrapper_full_tracking_with_synthetic_boundaries(tmp_path,synthetic_chain,monkeypatch):
    module,train,test,ids,manifest,components,config,calls=synthetic_chain
    data=tmp_path/'synthetic-data';data.mkdir()
    for name in ('train.json','ytrain.csv','test.json','ytest.csv','sample_submission.csv'):
        (data/name).write_text('EXPLICIT SYNTHETIC INPUT, NOT OFFICIAL DATA')
    before=module.data_manifest(data)
    monkeypatch.setattr(module,'validate_data',lambda _: (train,test,ids,before))
    monkeypatch.setattr(module,'validate_membership',lambda *a: [])
    monkeypatch.setattr(module,'validate_counts',lambda *a:dict(synthetic_fixture=True))
    monkeypatch.setattr(module,'cv_splits',lambda *a: [])
    monkeypatch.setattr(module.adapter,'load_components',lambda *a: components)
    tool=tmp_path/'codex';tool.write_text('synthetic executable')
    cli=dict(executable=str(tool),executable_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),version='0.163.0')
    monkeypatch.setattr(module.adapter,'codex_preflight',lambda:cli)
    def external(command,**kwargs):
        batch=Path(command[command.index('--output-last-message')+1]).parent
        dids=json.loads((batch/'prepared.json').read_text())['dialog_ids']
        (batch/'output.json').write_text(json.dumps({'predictions':[dict(dialog_id=d,p0=0.6,p1=0.4,reason='synthetic') for d in dids]}))
        kwargs['stdout'].write(json.dumps({'item':{'type':'agent_message'}})+'\n')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(module.adapter.subprocess,'run',external)
    report=module.run(data,tmp_path/'outputs','outer-synthetic','fresh')
    target=tmp_path/'outputs/outer-synthetic'
    receipt=json.loads((target/'tracking.json').read_text())
    assert receipt['status']=='FINISHED' and report['synthetic_fixture'] is True
    assert 'immutability.json' in (target/'index.html').read_text()
    immutable=json.loads((target/'immutability.json').read_text())
    assert immutable['all_unchanged'] is True and before==module.data_manifest(data)
    assert receipt['params']['requested_mode']=='fresh' and calls==['mock_native_no_fit']
    for forbidden in ('report.html','ENSEMBLE.md','README.md','tools/build_html.py'):
        assert forbidden not in immutable['source_before']
