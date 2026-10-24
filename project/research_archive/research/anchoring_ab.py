"""Matched no-upload A/B probe: prior-anchored versus independent LLM evidence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import log_loss

from botbench import file_sha, load_dataset, membership_folds, read_probabilities
from experiment import ROOT
from finalize import paired_report
from llm_bridge import PROMPT, infer as bridge_infer, prepare_batch, validate_output

UNANCHORED_PROMPT = PROMPT
for original, replacement in [
    ("do not overrule learned priors merely\nbecause a reply is short.",
     "do not assume a reply is human merely\nbecause it is short."),
    (' The statistical model prior is already strong and based\n'
     'on held-out models; change it only when the context adds evidence, not to average toward 0.5.',
     ' Make an independent assessment from the dialogue and known training references.'),
    ('Each payload includes prior probabilities from a trained classifier and nearest known TRAIN\nexamples.',
     'Each payload includes nearest known TRAIN examples. No trained-classifier probability is supplied.'),
]:
    if UNANCHORED_PROMPT.count(original) != 1:
        raise ValueError('anchored prompt changed; intervention no longer isolated')
    UNANCHORED_PROMPT = UNANCHORED_PROMPT.replace(original, replacement)


def independent_payload(payload):
    result = copy.deepcopy(payload)
    for dialogue in result:
        for side in dialogue['sides']:
            if 'prior_bot_probability' not in side:
                raise ValueError('control prior field missing before intervention')
            del side['prior_bot_probability']
    return result


def write_json(path, payload):
    with path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def closure():
    paths = [Path(__file__).resolve(), ROOT / 'botbench.py', ROOT / 'experiment.py',
             ROOT / 'llm_bridge.py', ROOT / 'finalize.py',
             ROOT / 'evidence/membership-v2.json', ROOT / 'data/train.json',
             ROOT / 'data/ytrain.csv', ROOT / 'runs/template-v1/oof.csv',
             ROOT / 'runs/final-v1/submission.csv',
             ROOT / 'evidence/anchoring-ab-v2-prereg.md']
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    frozen = closure()
    if frozen['runs/final-v1/submission.csv'] != '4bde02a00f00d009ad69c3aa967c00e468a41c0bcb2b90f65b93416abb3d4639':
        raise ValueError('incumbent artifact changed')
    rows = load_dataset(ROOT / 'data', 'train')
    membership = json.loads((ROOT / 'evidence/membership-v2.json').read_text())
    folds = membership_folds(rows, membership)
    eligible = sorted({r['dialog_id'] for r, f in zip(rows, folds) if f == 1})
    selected = np.random.default_rng(1729).choice(eligible, size=80, replace=False).tolist()
    training = [r for r, f in zip(rows, folds) if f in (0, 2, 3, 4)]
    priors = read_probabilities(ROOT / 'runs/template-v1/oof.csv', has_fold=True)
    for offset in range(0, 80, 40):
        chosen = set(selected[offset:offset+40])
        evaluation = [{k:v for k,v in r.items() if k != 'y'}
                      for r, f in zip(rows, folds) if f == 1 and r['dialog_id'] in chosen]
        control = out / 'anchored' / f'batch-{offset//40:02d}'
        prepare_batch(training, evaluation, priors, control)
        write_json(control / 'experiment.json', {'arm': 'anchored', 'source_closure': frozen})
        treatment = out / 'independent' / control.name
        treatment.mkdir(parents=True, exist_ok=False)
        payload = independent_payload(json.loads((control / 'input.json').read_text()))
        write_json(treatment / 'input.json', payload)
        shutil.copyfile(control / 'schema.json', treatment / 'schema.json')
        with (treatment / 'prompt.txt').open('x', encoding='utf-8') as f:
            f.write(UNANCHORED_PROMPT + '\nQUOTED_DATA_JSON:\n' + json.dumps(payload, ensure_ascii=False))
        prepared = json.loads((control / 'prepared.json').read_text())
        prepared['input_sha256'] = file_sha(treatment / 'input.json')
        prepared['prompt_sha256'] = file_sha(treatment / 'prompt.txt')
        write_json(treatment / 'prepared.json', prepared)
        write_json(treatment / 'experiment.json', {'arm': 'independent', 'source_closure': frozen})
    write_json(out / 'experiment.json', {'phase': 'fresh_development_fold1_screen',
        'selected_dialog_ids': selected, 'evaluated_participants': 160,
        'training_fold_ids': [0,2,3,4], 'excluded_confirmation': True,
        'primary_decoder': '0.5*template + 0.5*LLM', 'source_closure': frozen,
        'submitted': False})
    if closure() != frozen:
        raise ValueError('preparation source/input changed')
    print('PREPARED80dialogs per arm; fold1 labels excluded; no dispatch or upload', flush=True)


def infer(batch):
    receipt = json.loads((batch / 'experiment.json').read_text())
    if closure() != receipt['source_closure']:
        raise ValueError('A/B preparation-source/input preflight failed')
    bridge_infer(batch)
    if closure() != receipt['source_closure']:
        raise ValueError('A/B source/input changed during inference')


def read_arm(directory):
    values = {}
    for batch in sorted(directory.glob('batch-*')):
        e = json.loads((batch / 'experiment.json').read_text())
        if closure() != e['source_closure']:
            raise ValueError('A/B source closure mismatch')
        prepared = json.loads((batch / 'prepared.json').read_text())
        receipt = json.loads((batch / 'receipt.json').read_text())
        for filename, field in [('input.json','input_sha256'),('prompt.txt','prompt_sha256'),
                                ('output.json','output_sha256')]:
            if file_sha(batch / filename) != receipt[field]:
                raise ValueError('A/B inference artifact hash mismatch')
            if filename != 'output.json' and prepared[field] != receipt[field]:
                raise ValueError('prepared-to-inference binding mismatch')
        kinds = []
        for line in (batch / 'events.jsonl').read_text().splitlines():
            kind = json.loads(line).get('item', {}).get('type')
            if kind:
                kinds.append(kind)
                if kind not in {'agent_message','reasoning'}:
                    raise ValueError('unexpected tool event')
        if kinds != receipt['event_item_types']:
            raise ValueError('event log binding mismatch')
        records = validate_output(json.loads((batch / 'output.json').read_text()), prepared['dialog_ids'])
        for record in records:
            for side in (0,1):
                identifier = f'{record["dialog_id"]}_{side}'
                if identifier in values:
                    raise ValueError('duplicate A/B prediction ID')
                values[identifier] = record[f'p{side}']
    return values


def score(out):
    experiment = json.loads((out / 'experiment.json').read_text())
    control, independent = read_arm(out / 'anchored'), read_arm(out / 'independent')
    all_rows = load_dataset(ROOT / 'data','train')
    rows = [r for r in all_rows if r['dialog_id'] in set(experiment['selected_dialog_ids'])]
    expected = {r['ID'] for r in rows}
    if len(rows) != 160 or set(control) != expected or set(independent) != expected:
        raise ValueError('A/B membership or count differs')
    priors = read_probabilities(ROOT / 'runs/template-v1/oof.csv', has_fold=True)
    native = np.array([priors[r['ID']] for r in rows])
    anchored = np.array([control[r['ID']] for r in rows])
    unanchored = np.array([independent[r['ID']] for r in rows])
    a, b = .5*native+.5*anchored, .5*native+.5*unanchored
    comparison = paired_report(rows,a,b)
    report = {'phase':'fresh_fold1_development_screen','comparison':comparison,
        'raw_anchored_logloss':float(log_loss([r['y'] for r in rows],anchored)),
        'raw_independent_logloss':float(log_loss([r['y'] for r in rows],unanchored)),
        'native_logloss':float(log_loss([r['y'] for r in rows],native)),
        'slices':{},'source_closure':closure(),'submitted':False,
        'admit_larger_crossfit':comparison['delta_candidate_minus_base']<0 and
            comparison['group_bootstrap_95pct_interval'][1]<0}
    for name, mask in [('2-5_messages',np.array([len(r['messages'])<=5 for r in rows])),
                       ('6+_messages',np.array([len(r['messages'])>5 for r in rows]))]:
        selected=[r for r,m in zip(rows,mask) if m]
        report['slices'][name]=paired_report(selected,a[mask],b[mask])
    report['verdict']='PARTIAL_SCREEN_PASS' if report['admit_larger_crossfit'] else 'SCREEN_NOT_ADMITTED'
    write_json(out/'score.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_closure'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','infer','score'])
    parser.add_argument('path',type=Path)
    args=parser.parse_args()
    {'prepare':prepare,'infer':infer,'score':score}[args.action](args.path)
