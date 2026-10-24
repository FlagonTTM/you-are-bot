"""Validation, sealed-control preparation, and honest paired metric receipts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import log_loss

from botbench import file_sha, load_dataset, membership_folds
from experiment import ROOT, fit_predict, predict, provenance, write_predictions
from llm_bridge import validate_output


def validate_submission(path: Path, sample: Path) -> dict:
    with sample.open(newline='', encoding='utf-8') as f:
        expected = list(csv.DictReader(f))
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['ID', 'is_bot']:
            raise ValueError('submission header must be exactly ID,is_bot')
        rows = list(reader)
    if any(set(r) != {'ID', 'is_bot'} or r['ID'] is None or r['is_bot'] is None for r in rows):
        raise ValueError('submission rows must each have exactly two columns')
    ids = [r['ID'] for r in rows]
    if ids != [r['ID'] for r in expected] or len(ids) != len(set(ids)):
        raise ValueError('submission membership/order/uniqueness differs from official sample')
    try:
        probabilities = np.asarray([float(r['is_bot']) for r in rows])
    except (ValueError, TypeError):
        raise ValueError('submission probability is not numeric') from None
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('submission probabilities must be finite and in [0,1]')
    return {'rows': len(rows), 'sha256': file_sha(path),
            'min_probability': float(probabilities.min()),
            'max_probability': float(probabilities.max()),
            'sample_sha256': file_sha(sample)}


def loss_vector(y, p):
    p = np.clip(np.asarray(p, dtype=float), np.finfo(float).eps, 1-np.finfo(float).eps)
    y = np.asarray(y)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def paired_report(rows, base, candidate):
    y = np.asarray([r['y'] for r in rows])
    delta = loss_vector(y, candidate) - loss_vector(y, base)
    group_ids = sorted({r['group'] for r in rows})
    sums = np.asarray([delta[[r['group'] == g for r in rows]].sum() for g in group_ids])
    counts = np.asarray([sum(r['group'] == g for r in rows) for g in group_ids])
    rng = np.random.default_rng(1729)
    selected = rng.integers(0, len(group_ids), size=(3000, len(group_ids)))
    bootstrap = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {'base_logloss': float(log_loss(y, base, labels=[0, 1])),
            'candidate_logloss': float(log_loss(y, candidate, labels=[0, 1])),
            'delta_candidate_minus_base': float(delta.mean()),
            'group_bootstrap_95pct_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
            'participant_better': int((delta < -1e-12).sum()),
            'participant_worse': int((delta > 1e-12).sum()),
            'n_rows': len(rows), 'n_components': len(group_ids)}


def prepare_controls(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    source = provenance()
    rows = load_dataset(ROOT / 'data', 'train')
    membership = json.loads((ROOT / 'evidence/membership-v2.json').read_text())
    folds = membership_folds(rows, membership)
    training = [r for r, fold in zip(rows, folds) if fold >= 0]
    evaluation = [{k: v for k, v in r.items() if k != 'y'}
                  for r, fold in zip(rows, folds) if fold == -1]
    if {r['group'] for r in training} & {r['group'] for r in evaluation}:
        raise ValueError('confirmation contamination')
    artifacts = {}
    for name in ['behavior', 'template']:
        started = time.perf_counter()
        p, model = fit_predict(name, training, evaluation)
        path = out / f'{name}.joblib'
        joblib.dump(model, path)
        loaded = joblib.load(path)  # Only a locally created, hash-bound trusted model.
        replay = predict(name, loaded, evaluation)
        if not np.array_equal(p, replay):
            raise ValueError('checkpoint replay is not exact')
        write_predictions(out / f'{name}.csv', evaluation, p)
        artifacts[name] = {'model_sha256': file_sha(path),
                           'predictions_sha256': file_sha(out / f'{name}.csv'),
                           'seconds': time.perf_counter() - started,
                           'checkpoint_exact_replay': True}
        print('PREPARED_WITHOUT_SCORING',name,len(evaluation),flush=True)
    if provenance() != source:
        raise ValueError('source/input changed')
    receipt = {'train_IDs': [r['ID'] for r in training],
               'evaluation_IDs': [r['ID'] for r in evaluation],
               'confirmation_metrics_opened': False, 'provenance': source,
               'artifacts': artifacts, 'finalize_sha256': file_sha(Path(__file__))}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))


def llm_predictions(directory: Path, *, expected_bridge_sha=None):
    priors, probabilities, input_hashes = {}, {}, {}
    for batch in sorted(directory.glob('batch-*')):
        prepared = json.loads((batch / 'prepared.json').read_text())
        receipt = json.loads((batch / 'receipt.json').read_text())
        if prepared.get('evaluated_labels_provided') is not False:
            raise ValueError('batch lacks the no-evaluated-labels contract')
        if set(prepared['training_groups']) & set(prepared['evaluation_groups']):
            raise ValueError('batch reference/evaluation group overlap')
        if expected_bridge_sha is not None and prepared['bridge_sha256'] != expected_bridge_sha:
            raise ValueError('batch does not match the accepted generation bridge source')
        for field in ['input_sha256', 'prompt_sha256']:
            if receipt[field] != prepared[field]:
                raise ValueError('LLM prepared/receipt binding mismatch')
        for name, field in [('input.json', 'input_sha256'), ('prompt.txt', 'prompt_sha256'),
                            ('output.json', 'output_sha256')]:
            if file_sha(batch / name) != receipt[field]:
                raise ValueError('LLM receipt hash mismatch')
        actual_types = []
        for line in (batch / 'events.jsonl').read_text().splitlines():
            event = json.loads(line)
            kind = event.get('item', {}).get('type')
            if kind:
                actual_types.append(kind)
            if kind and kind not in {'reasoning', 'agent_message'}:
                raise ValueError('tool event in saved LLM inference')
        if actual_types != receipt['event_item_types']:
            raise ValueError('saved event log differs from inference receipt')
        results = validate_output(json.loads((batch / 'output.json').read_text()), prepared['dialog_ids'])
        data = {a['dialog_id']: a for a in json.loads((batch / 'input.json').read_text())}
        for result in results:
            for p in (0, 1):
                identifier = f'{result["dialog_id"]}_{p}'
                if identifier in probabilities:
                    raise ValueError('duplicate LLM result across batches')
                probabilities[identifier] = result[f'p{p}']
                sides = data[result['dialog_id']]['sides']
                priors[identifier] = next(s['prior_bot_probability'] for s in sides if s['participant_index'] == p)
        input_hashes[str(batch)] = receipt
    if not probabilities:
        raise ValueError('no LLM predictions')
    return priors, probabilities, input_hashes


def score_screen(directory: Path, output: Path):
    priors, llm, receipts = llm_predictions(directory)
    rows = [r for r in load_dataset(ROOT / 'data', 'train') if r['ID'] in llm]
    if set(llm) != {r['ID'] for r in rows} or len(rows) != 160:
        raise ValueError('screen membership/count mismatch')
    baseline = np.asarray([priors[r['ID']] for r in rows])
    remote = np.asarray([llm[r['ID']] for r in rows])
    candidate = .5 * baseline + .5 * remote
    comparison = paired_report(rows, baseline, candidate)
    report = {'phase': 'development_screen_only', 'comparison': comparison,
              'raw_llm_logloss': float(log_loss([r['y'] for r in rows], remote)),
              'admit_confirmation': comparison['delta_candidate_minus_base'] < 0,
              'recipe': '0.5*template + 0.5*llm', 'receipts': receipts,
              'confirmation_opened': False, 'submitted': False}
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ['comparison', 'raw_llm_logloss', 'admit_confirmation']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['controls', 'score-screen', 'validate'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'controls':
        prepare_controls(args.path)
    elif args.action == 'score-screen':
        score_screen(args.path, args.output)
    else:
        print(json.dumps(validate_submission(args.path, ROOT / 'data/sample_submission.csv'), indent=2))
