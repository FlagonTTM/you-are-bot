"""Bounded campaign phases; each output namespace is exclusive and replayable."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.metrics import log_loss

from botbench import file_sha, load_dataset, membership_folds, read_probabilities
from experiment import ROOT, fit_predict, predict, write_predictions
from finalize import llm_predictions, paired_report, validate_submission
from llm_bridge import prepare_batch


def read_probs(path):
    return read_probabilities(path)


def confirmation_parts():
    rows = load_dataset(ROOT / 'data', 'train')
    membership = json.loads((ROOT / 'evidence/membership-v2.json').read_text())
    folds = membership_folds(rows, membership)
    return ([r for r, fold in zip(rows, folds) if fold >= 0],
            [r for r, fold in zip(rows, folds) if fold == -1])


def batches(training, evaluation, priors, out):
    dialogs = sorted({r['dialog_id'] for r in evaluation})
    for offset in range(0, len(dialogs), 40):
        chosen = set(dialogs[offset:offset+40])
        target = [{k: v for k, v in r.items() if k != 'y'}
                  for r in evaluation if r['dialog_id'] in chosen]
        prepare_batch(training, target, priors, out / f'batch-{offset//40:02d}')
    print('PREPARED',len(dialogs),'dialogs',flush=True)


def prepare_confirmation(out):
    screen = json.loads((ROOT / 'runs/llm-screen-v1/score.json').read_text())
    if not screen['admit_confirmation']:
        raise ValueError('LLM failed screen; confirmation forbidden')
    training, evaluation = confirmation_parts()
    prior = read_probs(ROOT / 'runs/confirmation-controls-v1/template.csv')
    if set(prior) != {r['ID'] for r in evaluation}:
        raise ValueError('confirmation prior identity mismatch')
    batches(training, evaluation, prior, out)


def score_confirmation(directory, output):
    training, rows = confirmation_parts()
    behavior = read_probs(ROOT / 'runs/confirmation-controls-v1/behavior.csv')
    template = read_probs(ROOT / 'runs/confirmation-controls-v1/template.csv')
    prior, llm, receipts = llm_predictions(directory)
    expected = {r['ID'] for r in rows}
    if any(set(m) != expected for m in (behavior, template, prior, llm)):
        raise ValueError('confirmation must cover exact full frozen set')
    ids = [r['ID'] for r in rows]
    base = np.asarray([behavior[i] for i in ids])
    local = np.asarray([template[i] for i in ids])
    remote = np.asarray([llm[i] for i in ids])
    if not np.array_equal(local, [prior[i] for i in ids]):
        raise ValueError('LLM prior differs from saved checkpoint predictions')
    candidate = .5 * local + .5 * remote
    against_behavior = paired_report(rows, base, candidate)
    against_template = paired_report(rows, local, candidate)
    improvement = (against_behavior['delta_candidate_minus_base'] < 0 and
                   against_template['delta_candidate_minus_base'] < 0)
    robust = improvement and against_template['group_bootstrap_95pct_interval'][1] < 0
    report = {'phase': 'single_confirmation', 'recipe': '0.5*template + 0.5*llm',
              'against_behavior': against_behavior, 'against_template': against_template,
              'template_vs_behavior': paired_report(rows, base, local),
              'raw_llm_logloss': float(log_loss([r['y'] for r in rows], remote)),
              'verdict': 'VALIDATED' if robust else ('PARTIAL' if improvement else 'INVALIDATED'),
              'admit_final_llm': bool(robust), 'receipts': receipts, 'submitted': False}
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='receipts'},indent=2),flush=True)


def prepare_test(out):
    decision = json.loads((ROOT / 'runs/confirmation-decision-v1.json').read_text())
    if not decision['admit_final_llm']:
        raise ValueError('semantic classifier did not pass confirmation gate')
    out.mkdir(parents=True, exist_ok=False)
    training = load_dataset(ROOT / 'data', 'train')
    evaluation = load_dataset(ROOT / 'data', 'test')
    started = time.perf_counter()
    prior, model = fit_predict('template', training, evaluation)
    joblib.dump(model, out / 'template-fulltrain.joblib')
    replay = predict('template', joblib.load(out / 'template-fulltrain.joblib'), evaluation)
    if not np.array_equal(prior, replay):
        raise ValueError('final checkpoint did not exactly replay')
    write_predictions(out / 'template-prior.csv', evaluation, prior)
    validate_submission(out / 'template-prior.csv', ROOT / 'data/sample_submission.csv')
    priors = {r['ID']: float(p) for r, p in zip(evaluation, prior)}
    batches(training, evaluation, priors, out / 'llm')
    report = {'training_rows': len(training), 'test_rows': len(evaluation),
              'seconds': time.perf_counter() - started,
              'checkpoint_sha256': file_sha(out / 'template-fulltrain.joblib'),
              'checkpoint_exact_replay': True,
              'prior_sha256': file_sha(out / 'template-prior.csv'),
              'source_sha256': {p.name:file_sha(p) for p in ROOT.glob('*.py')},
              'decision_sha256': file_sha(ROOT / 'runs/confirmation-decision-v1.json')}
    snapshot = out / 'source-snapshot'
    snapshot.mkdir()
    for name in report['source_sha256']:
        shutil.copyfile(ROOT / name, snapshot / name)
    (out / 'prepared.json').write_text(json.dumps(report,indent=2))


def verify_final_run(directory):
    prepared = json.loads((directory / 'prepared.json').read_text())
    decision_path = ROOT / 'runs/confirmation-decision-v1.json'
    checks = [(decision_path, prepared['decision_sha256']),
              (directory / 'template-fulltrain.joblib', prepared['checkpoint_sha256']),
              (directory / 'template-prior.csv', prepared['prior_sha256'])]
    sources = prepared['source_sha256']
    required_sources = {'botbench.py', 'experiment.py', 'llm_bridge.py', 'finalize.py', 'campaign.py'}
    if set(sources) != required_sources:
        raise ValueError('generation source inventory is incomplete')
    checks.extend((directory / 'source-snapshot' / name, digest) for name, digest in sources.items())
    control = json.loads((ROOT / 'runs/confirmation-controls-v1/receipt.json').read_text())
    data_hashes = {name: digest for name, digest in control['provenance']['sha256'].items()
                   if name.startswith('data/')}
    if set(data_hashes) != {'data/train.json', 'data/test.json', 'data/ytrain.csv',
                            'data/ytest.csv', 'data/sample_submission.csv'}:
        raise ValueError('official input inventory is incomplete')
    checks.extend((ROOT / name, digest) for name, digest in data_hashes.items())
    for path, digest in checks:
        if not isinstance(digest, str) or len(digest) != 64 or file_sha(path) != digest:
            raise ValueError(f'final provenance hash mismatch: {path}')
    decision = json.loads(decision_path.read_text())
    if decision.get('admit_final_llm') is not True or decision.get('recipe') != '0.5*template + 0.5*llm':
        raise ValueError('final recipe does not have the frozen confirmation approval')
    return {'generation_sources_verified': len(sources), 'official_inputs_verified': len(data_hashes),
            'generation_bridge_sha256': sources['llm_bridge.py'],
            'generation_manifest_sha256': file_sha(directory / 'prepared.json'),
            'verification_source_sha256': {p.name: file_sha(p) for p in ROOT.glob('*.py')}}


def assemble(directory):
    verification = verify_final_run(directory)
    priors, remote, receipts = llm_predictions(
        directory / 'llm', expected_bridge_sha=verification['generation_bridge_sha256'])
    test = load_dataset(ROOT / 'data', 'test')
    if set(priors) != {r['ID'] for r in test} or set(remote) != set(priors):
        raise ValueError('final LLM output does not cover exact test membership')
    saved_prior = read_probs(directory / 'template-prior.csv')
    if priors != saved_prior:
        raise ValueError('final prior mismatch')
    p = [.5 * priors[r['ID']] + .5 * remote[r['ID']] for r in test]
    path = directory / 'submission.csv'
    write_predictions(path, test, p)
    report = validate_submission(path, ROOT / 'data/sample_submission.csv')
    report.update({'recipe': '0.5*template + 0.5*llm', 'llm_batches': len(receipts),
                   'generation_verification': verification,
                   'submitted': False, 'local_verification_only': True})
    (directory / 'submission-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare-confirmation','score-confirmation','prepare-test','assemble'])
    parser.add_argument('path',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='prepare-confirmation': prepare_confirmation(args.path)
    elif args.action=='score-confirmation': score_confirmation(args.path,args.output)
    elif args.action=='prepare-test': prepare_test(args.path)
    else: assemble(args.path)
