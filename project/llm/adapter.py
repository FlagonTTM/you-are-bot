"""Blind LLM boundary. Only project-local code; no credentials or old entrypoints."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import math
from pathlib import Path
import json
import shutil
import subprocess
from types import SimpleNamespace

from guards import sha256, strict_json
from llm import bridge

ROOT = Path(__file__).resolve().parent
SYMBOLS = {'MODEL', 'PROMPT', 'validate_output', 'first_text', 'prepare_batch'}


def verify_sources():
    config = strict_json(ROOT / 'config.json')
    symbols = []
    for filename, key in [('bridge.py', 'bridge_sha256'), ('historical_bridge.py', 'historical_bridge_sha256')]:
        path = ROOT / filename
        if path.is_symlink() or sha256(path) != config[key]:
            raise ValueError('pinned bridge source SHA mismatch')
        tree = ast.parse(path.read_bytes())
        selected = {}
        for node in tree.body:
            name = node.name if isinstance(node, ast.FunctionDef) else (
                node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
            if name in SYMBOLS:
                selected[name] = ast.dump(node, include_attributes=False)
        symbols.append(selected)
    if set(symbols[0]) != SYMBOLS or symbols[0] != symbols[1]:
        raise ValueError('scored prompt/retrieval/decoder AST parity mismatch')
    return dict(all_scientific_ast_equal=True, scientific_symbols=sorted(SYMBOLS),
                bridge_sha256=config['bridge_sha256'], historical_bridge_sha256=config['historical_bridge_sha256'],
                scientific_ast_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in symbols[0].items()})


def probability(value, *, strict=False):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1 or (strict and not 0 < value < 1):
        raise ValueError('invalid probability')
    return float(value)


def load_components(path, expected_ids, expected_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('component cache SHA mismatch')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
    if reader.fieldnames != ['ID', 'catboost', 'llm']:
        raise ValueError('component CSV schema mismatch')
    result = {}
    for row in reader:
        if set(row) != {'ID', 'catboost', 'llm'} or any(v is None for v in row.values()):
            raise ValueError('component CSV width mismatch')
        if not row['ID'] or row['ID'] in result:
            raise ValueError('duplicate/empty component ID')
        result[row['ID']] = {key: probability(float(row[key]), strict=(key == 'llm')) for key in ('catboost', 'llm')}
    if list(result) != list(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError('component ID/order/cardinality mismatch')
    return result


def prepare_batches(training, evaluation, priors, out):
    verify_sources()
    if any('y' in r or 'is_bot' in r for r in evaluation):
        raise ValueError('evaluated target refused')
    if not training or any(type(r.get('y')) not in (int, float) or r['y'] not in (0, 1) for r in training):
        raise ValueError('training labels required')
    if {r['group'] for r in training} & {r['group'] for r in evaluation}:
        raise ValueError('training/evaluation group overlap')
    ids = [r['ID'] for r in evaluation]
    if len(ids) != len(set(ids)) or set(priors) != set(ids):
        raise ValueError('evaluation/prior ID mismatch')
    for p in priors.values():
        probability(p)
    dialogs = sorted({r['dialog_id'] for r in evaluation})
    if not dialogs or len(dialogs) > 400:
        raise ValueError('dialogue budget exceeded')
    for did in dialogs:
        sides = [r['participant_index'] for r in evaluation if r['dialog_id'] == did]
        if sorted(sides) != [0, 1]:
            raise ValueError('expected exactly two distinct participants')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    paths = []
    # Exact campaign.batches rule: sorted dialogue chunks; original row order
    # inside chunks. Do NOT sort target participants by sample order.
    for offset in range(0, len(dialogs), 40):
        chosen = set(dialogs[offset:offset+40])
        target = [r for r in evaluation if r['dialog_id'] in chosen]
        path = out / f'batch-{offset//40:02d}'
        bridge.prepare_batch(training, target, priors, path)
        (path / 'boundary-lock.json').write_text(json.dumps(batch_lock(path), indent=2), encoding='utf-8')
        paths.append(path)
    return paths


def bind_cache(batches, manifest, priors, components):
    if set(priors) != set(components) or any(priors[i] != components[i]['catboost'] for i in priors):
        raise ValueError('new native prior differs exactly from historical cache')
    if {p.name for p in batches} != set(manifest['source_batch_digests']):
        raise ValueError('cached batch inventory mismatch')
    checked = {}
    for batch in batches:
        actual = {key: sha256(batch / filename) for key, filename in [('input_sha256', 'input.json'), ('prompt_sha256', 'prompt.txt')]}
        for key, digest in actual.items():
            if digest != manifest['source_batch_digests'][batch.name][key]:
                raise ValueError('historical cache ' + key + ' binding mismatch')
        checked[batch.name] = actual
    return dict(all_exact=True, prior_comparison='exact Python float equality; no tolerance', batches=checked)


def codex_preflight():
    executable = shutil.which('codex')
    if executable is None:
        raise FileNotFoundError('Codex CLI 0.163.0 is required for fresh mode')
    executable = str(Path(executable).resolve(strict=True))
    version = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=15)
    if version.returncode or version.stdout.strip() != 'codex-cli 0.163.0':
        raise ValueError('fresh mode requires exact Codex CLI 0.163.0')
    login = subprocess.run([executable, 'login', 'status'], capture_output=True, text=True, timeout=15)
    status = (login.stdout + login.stderr).strip()
    if login.returncode or status != 'Logged in using ChatGPT':
        raise ValueError('fresh mode requires existing ChatGPT Codex authentication; use CLI login separately')
    return dict(executable=executable, executable_sha256=sha256(Path(executable)), version='0.163.0',
                authentication='Logged in using ChatGPT', credential_files_read_by_pipeline=False)


def batch_lock(batch):
    result = {}
    for name in ('input.json', 'prompt.txt', 'schema.json', 'prepared.json'):
        path = batch / name
        if path.is_symlink():
            raise ValueError('batch symlink refused')
        result[name] = sha256(path)
    return result


def fresh_infer(batch, cli, check_immutable, *, lock=None):
    """Execute audited infer once, with additional raw/source/schema/CLI guards.

    The copied bridge is byte-identical to the audited guard-corrected source.
    Its subprocess namespace is scoped to this serial call, NOT monkeypatching
    Python's shared subprocess module. Scientific preparation is untouched.
    """
    if any((batch / name).exists() for name in ('output.json', 'events.jsonl', 'stderr.log', 'receipt.json')):
        raise ValueError('existing inference; never automatically resend')
    lock = strict_json(batch / 'boundary-lock.json') if lock is None else lock
    verify_sources()
    def guard():
        check_immutable()
        if batch_lock(batch) != lock:
            raise ValueError('source/input/prompt/schema/prepared preflight mismatch')
        if sha256(Path(cli['executable'])) != cli['executable_sha256']:
            raise ValueError('CLI executable changed')
    guard()
    sandbox = batch.parent / 'inference-sandbox'
    sandbox.mkdir(exist_ok=True)
    old_root, old_subprocess = bridge.ROOT, bridge.subprocess
    dispatched = []
    def dispatch(command, **kwargs):
        guard()
        # The original audited argv is retained, only resolving its executable.
        if command[0] != 'codex' or command[1] != 'exec':
            raise ValueError('unexpected inference command')
        actual = [cli['executable'], *command[1:]]
        kwargs['timeout'] = 601
        dispatched.append(actual)
        try:
            return subprocess.run(actual, **kwargs)
        finally:
            guard()
    try:
        bridge.ROOT = sandbox
        bridge.subprocess = SimpleNamespace(run=dispatch)
        bridge.infer(batch)
    finally:
        bridge.ROOT, bridge.subprocess = old_root, old_subprocess
    guard()
    prepared = strict_json(batch / 'prepared.json')
    predictions = bridge.validate_output(strict_json(batch / 'output.json'), prepared['dialog_ids'])
    receipt = dict(boundary_before=lock, boundary_after=batch_lock(batch),
                   all_unchanged=True, timeout_seconds=601, command=dispatched[0],
                   cli_version=cli['version'], cli_executable_sha256=cli['executable_sha256'],
                   mode='fresh', automatic_resend=False)
    with (batch / 'boundary-receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    return predictions
