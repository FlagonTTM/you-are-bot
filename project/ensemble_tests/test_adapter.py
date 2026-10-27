"""Tiny artificial dialogues only: no official fit, account or inference calls."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def test_scientific_prompt_and_retrieval_match_scored_source():
    assert (ROOT / 'llm/adapter.py').is_file(), 'new blind LLM adapter is missing'
    adapter = importlib.import_module('llm.adapter')
    parity = adapter.verify_sources()
    assert parity['all_scientific_ast_equal'] is True
    assert set(parity['scientific_symbols']) == {'PROMPT', 'MODEL', 'first_text', 'prepare_batch', 'validate_output'}
    assert parity['historical_bridge_sha256'] == 'dc95650cc3cf3081e87c2849a03d3f332c6b43e0fbdca5e95167d57253b118d6'


@pytest.fixture
def artificial_rows():
    from uuid import UUID
    def dialogue(n, train):
        did = str(UUID(int=n))
        messages = [dict(text='человек привет ' + str(n), participant_index=0),
                    dict(text='script robot reply ' + str(n), participant_index=1)]
        return [dict(ID=f'{did}_{p}', dialog_id=did, participant_index=p, group=str(n), messages=messages,
                     own_text=messages[p]['text'], other_text=messages[1-p]['text'], **({'y': p} if train else {}))
                for p in (0, 1)]
    return dialogue(2, True) + dialogue(1, True), dialogue(102, False) + dialogue(101, False)


def test_cached_outputs_require_exact_new_priors_and_payload_digests(tmp_path, artificial_rows):
    from llm import adapter
    assert hasattr(adapter, 'prepare_batches'), 'blind batch stage missing'
    training, evaluation = artificial_rows
    priors = {r['ID']: 0.25 for r in evaluation}
    batches = adapter.prepare_batches(training, evaluation, priors, tmp_path / 'batches')
    # Sorted chunks retain original evaluation order within each chunk, like campaign.batches.
    prepared = json.loads((batches[0] / 'prepared.json').read_text())
    assert prepared['evaluation_ids'] == list(priors)
    assert prepared['training_ids'] == [r['ID'] for r in training]
    payload = json.loads((batches[0] / 'input.json').read_text())
    assert all('y' not in d for d in payload)
    manifest = {'source_batch_digests': {'batch-00': {k: prepared[k] for k in ('input_sha256', 'prompt_sha256')}}}
    components = {key: {'catboost': p, 'llm': 0.6} for key, p in priors.items()}
    assert adapter.bind_cache(batches, manifest, priors, components)['all_exact'] is True
    components[next(iter(priors))]['catboost'] += 1e-15
    with pytest.raises(ValueError, match='prior'):
        adapter.bind_cache(batches, manifest, priors, components)
    components[next(iter(priors))]['catboost'] = 0.25
    (batches[0] / 'prompt.txt').write_text('tampered prompt')
    with pytest.raises(ValueError, match='prompt'):
        adapter.bind_cache(batches, manifest, priors, components)


@pytest.mark.parametrize('bad', ['duplicate', 'nan', 'bool', 'width', 'order', 'count', 'schema', 'hash'])
def test_component_csv_rejects_invalid_cache(tmp_path, bad):
    from llm import adapter
    assert hasattr(adapter, 'load_components'), 'strict component loader missing'
    text = 'ID,catboost,llm\r\na,0.25,0.6\r\nb,0.25,0.6\r\n'
    expected = ['a', 'b']
    text = {'duplicate': text.replace('b,', 'a,'), 'nan': text.replace('0.6', 'nan'),
            'bool': text.replace('0.6', 'True'), 'width': text.replace('b,0.25,0.6', 'b,0.25,0.6,0'),
            'schema': text.replace('catboost', 'native')}.get(bad, text)
    if bad == 'order': expected.reverse()
    if bad == 'count': expected.append('c')
    path = tmp_path / 'components.csv'
    path.write_bytes(text.encode())
    digest = '0' * 64 if bad == 'hash' else hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        adapter.load_components(path, expected, digest)


def test_test_targets_and_group_overlap_are_rejected_before_preparation(tmp_path, artificial_rows):
    from llm import adapter
    training, evaluation = artificial_rows
    priors = {r['ID']: 0.25 for r in evaluation}
    evaluation[0]['y'] = 1
    with pytest.raises(ValueError, match='target'):
        adapter.prepare_batches(training, evaluation, priors, tmp_path / 'target')
    del evaluation[0]['y']
    evaluation[0]['group'] = training[0]['group']
    with pytest.raises(ValueError, match='overlap'):
        adapter.prepare_batches(training, evaluation, priors, tmp_path / 'overlap')


@pytest.mark.parametrize('case', ['version', 'auth', 'missing'])
def test_fresh_cli_prerequisites_fail_without_any_model_call(monkeypatch, tmp_path, case):
    from llm import adapter
    assert hasattr(adapter, 'codex_preflight'), 'fresh CLI preflight missing'
    import subprocess
    tool = tmp_path / 'codex'
    tool.write_text('synthetic executable')
    calls = []
    monkeypatch.setattr(adapter.shutil, 'which', lambda _: None if case == 'missing' else str(tool))
    def fake_run(command, **kwargs):
        calls.append(command)
        assert 'exec' not in command
        if '--version' in command:
            return subprocess.CompletedProcess(command, 0, 'codex-cli 0.1.0' if case == 'version' else 'codex-cli 0.163.0', '')
        return subprocess.CompletedProcess(command, 1, '', 'Not logged in')
    monkeypatch.setattr(adapter.subprocess, 'run', fake_run)
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.codex_preflight()
    assert not any('exec' in c for c in calls)


def test_fresh_guard_uses_exact_checked_prompt_no_tools_and_refuses_resend(monkeypatch, tmp_path, artificial_rows):
    from llm import adapter
    assert hasattr(adapter, 'fresh_infer'), 'fresh guarded dispatch missing'
    import subprocess
    training, evaluation = artificial_rows
    priors = {r['ID']: 0.25 for r in evaluation}
    batch = adapter.prepare_batches(training, evaluation, priors, tmp_path / 'fresh')[0]
    tool = tmp_path / 'codex'; tool.write_text('synthetic executable')
    cli = dict(executable=str(tool), executable_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(), version='0.163.0')
    checks, calls = [], []
    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs['input'].encode() == (batch / 'prompt.txt').read_bytes()
        assert kwargs['timeout'] == 601
        assert command[0] == str(tool)
        assert '--ignore-user-config' in command and '--ephemeral' in command
        assert kwargs['cwd'] != adapter.ROOT
        assert all(flag in command for flag in ('shell_tool', 'browser_use', 'computer_use', 'plugins', 'hooks'))
        dids = json.loads((batch / 'prepared.json').read_text())['dialog_ids']
        (batch / 'output.json').write_text(json.dumps({'predictions': [dict(dialog_id=d, p0=0.6, p1=0.4, reason='synthetic') for d in dids]}))
        kwargs['stdout'].write(json.dumps({'item': {'type': 'agent_message'}}) + '\n')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(adapter.subprocess, 'run', fake_run)
    actual = adapter.fresh_infer(batch, cli, lambda: checks.append('check'))
    assert len(actual) == 2 and len(checks) >= 2 and len(calls) == 1
    with pytest.raises(ValueError, match='existing'):
        adapter.fresh_infer(batch, cli, lambda: None)
    assert len(calls) == 1


@pytest.mark.parametrize('target', ['prompt.txt', 'input.json', 'schema.json', 'prepared.json', 'source'])
def test_fresh_refuses_changed_inputs_before_dispatch(monkeypatch, tmp_path, artificial_rows, target):
    from llm import adapter
    assert hasattr(adapter, 'fresh_infer'), 'fresh guarded dispatch missing'
    training, evaluation = artificial_rows
    batches = adapter.prepare_batches(training, evaluation, {r['ID']: 0.25 for r in evaluation}, tmp_path / 'blocked')
    batch = batches[0]
    lock = adapter.batch_lock(batch)
    if target != 'source':
        (batch / target).write_text('changed bytes')
    def check():
        if target == 'source': raise ValueError('source changed')
    monkeypatch.setattr(adapter.subprocess, 'run', lambda *a, **k: pytest.fail('dispatch must be blocked'))
    with pytest.raises(ValueError):
        adapter.fresh_infer(batch, {}, check, lock=lock)
    assert not (batch / 'events.jsonl').exists()


def test_batch_chunks_sort_dialogues_but_keep_original_row_order(tmp_path, artificial_rows):
    from llm import adapter
    from uuid import UUID
    training, base = artificial_rows
    evaluation=[]
    for n in reversed(range(101,142)):
        did=str(UUID(int=n))
        evaluation.extend({**r,'ID':did+'_'+str(r['participant_index']),'dialog_id':did,'group':did} for r in base[:2])
    batches=adapter.prepare_batches(training,evaluation,{r['ID']:0.25 for r in evaluation},tmp_path/'chunked')
    assert len(batches)==2
    first=json.loads((batches[0]/'prepared.json').read_text())['evaluation_ids']
    second=json.loads((batches[1]/'prepared.json').read_text())['evaluation_ids']
    assert first==[r['ID'] for r in evaluation if r['dialog_id']!=str(UUID(int=141))]
    assert second==[str(UUID(int=141))+'_0',str(UUID(int=141))+'_1']


@pytest.mark.parametrize('case', ['post_input_change','post_source_change','tool_event','timeout','events_existing'])
def test_fresh_failed_attempt_is_not_resent(monkeypatch,tmp_path,artificial_rows,case):
    from llm import adapter
    import subprocess
    training,evaluation=artificial_rows
    batch=adapter.prepare_batches(training,evaluation,{r['ID']:0.25 for r in evaluation},tmp_path/'attempt')[0]
    tool=tmp_path/'codex'; tool.write_text('synthetic executable')
    cli=dict(executable=str(tool),executable_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),version='0.163.0')
    called=[]; changed=[]
    def check():
        if changed and case=='post_source_change': raise ValueError('source changed after return')
    def external(command,**kwargs):
        called.append(command)
        if case=='timeout': raise subprocess.TimeoutExpired(command,601)
        dids=json.loads((batch/'prepared.json').read_text())['dialog_ids']
        (batch/'output.json').write_text(json.dumps({'predictions':[dict(dialog_id=d,p0=0.6,p1=0.4,reason='synthetic') for d in dids]}))
        if case=='post_input_change': (batch/'input.json').write_text('changed after dispatch')
        changed.append(True)
        kwargs['stdout'].write(json.dumps({'item':{'type':'command_execution' if case=='tool_event' else 'agent_message'}})+'\n')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(adapter.subprocess,'run',external)
    if case=='events_existing': (batch/'events.jsonl').write_text('existing ambiguous attempt')
    with pytest.raises((ValueError,subprocess.TimeoutExpired)):
        adapter.fresh_infer(batch,cli,check)
    assert len(called)==(0 if case=='events_existing' else 1)
    with pytest.raises(ValueError,match='existing'):
        adapter.fresh_infer(batch,cli,check)
    assert len(called)==(0 if case=='events_existing' else 1)
