"""One fixed probe retaining incumbent template features plus native text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from catboost import CatBoostClassifier

import experiment
from botbench import file_sha, load_dataset, read_probabilities
from finalize import paired_report

MODEL_NAME = 'template_text'


def combined_input(rows):
    strong = experiment.model_input(rows, 'template')
    own, other = experiment.text_columns(rows)
    n = strong.shape[1]
    result = np.empty((len(rows), n + 2), dtype=object)
    result[:, :n] = strong
    result[:, n] = own
    result[:, n + 1] = other
    return result, list(range(n - 14, n)), [n, n + 1]


def combined_fit_predict(model_name, training, evaluation):
    if model_name != MODEL_NAME:
        raise ValueError('wrong experiment model contract')
    xtrain, cat, text = combined_input(training)
    model = CatBoostClassifier(**experiment.BEHAVIOR_CONFIG, cat_features=cat, text_features=text)
    model.fit(xtrain, [r['y'] for r in training])
    xeval, eval_cat, eval_text = combined_input(evaluation)
    if cat != eval_cat or text != eval_text:
        raise ValueError('combined input schema mismatch')
    return model.predict_proba(xeval)[:, 1], model


def run(out):
    root = experiment.ROOT
    baseline_source = json.loads((root / 'evidence/guard-correction-v2.json').read_text())['new_verification_source']
    if any(file_sha(root / n) != h for n, h in baseline_source.items()):
        raise ValueError('incumbent source changed')
    base_provenance = experiment.provenance

    def bounded_provenance():
        p = base_provenance()
        for path in [Path(__file__).resolve(), root / 'evidence/template-text-v2-prereg.md',
                     root / 'runs/template-v1/oof.csv', root / 'runs/template-v1/report.json']:
            p['sha256'][str(path.relative_to(root))] = file_sha(path)
        return p

    # Shared project runner, only process-local hooks; no incumbent source writes.
    with patch.object(experiment, 'fit_predict', combined_fit_predict), \
         patch.object(experiment, 'provenance', bounded_provenance):
        experiment.run(MODEL_NAME, out)
    if any(file_sha(root / n) != h for n, h in baseline_source.items()):
        raise ValueError('incumbent source changed during probe')
    prior = read_probabilities(root / 'runs/template-v1/oof.csv', has_fold=True)
    candidate = read_probabilities(out / 'oof.csv', has_fold=True)
    if set(prior) != set(candidate):
        raise ValueError('new/control OOF IDs differ')
    rows = [r for r in load_dataset(root / 'data','train') if r['ID'] in candidate]
    a = np.asarray([prior[r['ID']] for r in rows])
    b = np.asarray([candidate[r['ID']] for r in rows])
    comparison = paired_report(rows,a,b)
    report = {'model': MODEL_NAME, 'comparison':comparison,
              'changed_binary_decisions':int(((a>=.5)!=(b>=.5)).sum()),
              'admit_transfer':comparison['delta_candidate_minus_base']<0 and
                  comparison['group_bootstrap_95pct_interval'][1]<0,
              'slices':{},'submitted':False,'incumbent_unchanged':True}
    for name,mask in [('2-5_messages',np.asarray([len(r['messages'])<=5 for r in rows])),
                      ('6+_messages',np.asarray([len(r['messages'])>5 for r in rows]))]:
        selected=[r for r,m in zip(rows,mask) if m]
        report['slices'][name]=paired_report(selected,a[mask],b[mask])
    with (out/'comparison.json').open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.out)
