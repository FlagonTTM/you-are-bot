"""One shared, immutable-output runner; no external APIs, no submissions."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import joblib
from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from botbench import file_sha, load_dataset, membership_folds, numeric_features

ROOT = Path(__file__).resolve().parent
BEHAVIOR_CONFIG = dict(iterations=1000, depth=5, learning_rate=0.04,
                       l2_leaf_reg=5, loss_function='Logloss', random_seed=1729,
                       thread_count=4, verbose=False, allow_writing_files=False)
LEXICAL_CONFIG = {'C': 2.0, 'solver': 'liblinear', 'max_iter': 500,
                  'random_state': 1729}


def clean_text(text: str) -> str:
    # Preserve serialization syntax, never learn embedded identity values.
    masked = re.sub(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b',
                    ' UUIDVALUE ', text)
    return masked[:12000] or 'EMPTYMESSAGE'


def text_columns(rows):
    return [[clean_text(r[key]) for r in rows] for key in ('own_text', 'other_text')]


def model_input(rows, model_name):
    numeric, names = numeric_features(rows)
    if model_name == 'behavior':
        return numeric
    if model_name == 'template':
        extra, categories = [], []
        for r in rows:
            nums, cats = [], []
            for side in (r['participant_index'], 1 - r['participant_index']):
                own = [m['text'] for m in r['messages'] if m['participant_index'] == side]
                n = max(1, len(own))
                nums.extend([sum(t.startswith(' ') for t in own) / n,
                             sum(t.startswith('\n') for t in own) / n,
                             sum(t.endswith('\n') for t in own) / n,
                             sum('UUID(' in t for t in own) / n])
                previous, repeat_echo, compared, token_overlap = None, 0, 0, []
                for m in r['messages']:
                    t = ' '.join(m['text'].casefold().split())
                    if m['participant_index'] != side:
                        previous = t
                    elif previous:
                        compared += 1
                        repeat_echo += bool(t) and not t.replace(previous, '').strip()
                        a, b = set(t.split()), set(previous.split())
                        token_overlap.append(len(a & b) / max(1, len(a | b)))
                nums.extend([repeat_echo / max(1, compared),
                             float(np.mean(token_overlap or [0]))])
                early = [clean_text(t) for t in own[:3]]
                early += ['MISSINGMESSAGE'] * (3 - len(early))
                cats.extend([t[:256] for t in early])
                cats.extend([' '.join(t.casefold().split())[:256] for t in early])
                cats.append(' '.join(early[0].casefold().split()[:3]))
            extra.append(nums)
            categories.append(cats)
        continuous = np.column_stack([numeric, np.asarray(extra)])
        matrix = np.empty((len(rows), continuous.shape[1] + 14), dtype=object)
        matrix[:, :-14] = continuous
        matrix[:, -14:] = categories
        return matrix
    texts = text_columns(rows)
    matrix = np.empty((len(rows), numeric.shape[1] + 2), dtype=object)
    matrix[:, :-2] = numeric
    matrix[:, -2] = texts[0]
    matrix[:, -1] = texts[1]
    return matrix


class LexicalModel:
    def fit(self, rows):
        own, other = text_columns(rows)
        self.own_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                       min_df=2, max_features=80000, sublinear_tf=True)
        self.own_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                       max_features=30000, sublinear_tf=True)
        self.other_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                         min_df=2, max_features=30000, sublinear_tf=True)
        numeric, _ = numeric_features(rows)
        transformed = np.sign(numeric) * np.log1p(np.abs(numeric))
        self.scaler = StandardScaler().fit(transformed)
        x = hstack([self.own_char.fit_transform(own), self.own_word.fit_transform(own),
                    self.other_char.fit_transform(other),
                    csr_matrix(self.scaler.transform(transformed) / 3)], format='csr')
        self.model = LogisticRegression(**LEXICAL_CONFIG).fit(x, [r['y'] for r in rows])
        return self

    def predict_proba(self, rows):
        own, other = text_columns(rows)
        numeric, _ = numeric_features(rows)
        transformed = np.sign(numeric) * np.log1p(np.abs(numeric))
        x = hstack([self.own_char.transform(own), self.own_word.transform(own),
                    self.other_char.transform(other),
                    csr_matrix(self.scaler.transform(transformed) / 3)], format='csr')
        return self.model.predict_proba(x)


def predict(model_name, model, rows):
    if model_name == 'lexical':
        return model.predict_proba(rows)[:, 1]
    return model.predict_proba(model_input(rows, model_name))[:, 1]


def fit_predict(model_name: str, training: list[dict], evaluation: list[dict]):
    if model_name == 'lexical':
        model = LexicalModel().fit(training)
        return predict(model_name, model, evaluation), model
    if model_name not in {'behavior', 'hybrid', 'template'}:
        raise ValueError('unregistered model')
    xtrain = model_input(training, model_name)
    extra = {'text_features': [xtrain.shape[1] - 2, xtrain.shape[1] - 1]} if model_name == 'hybrid' else {}
    if model_name == 'template':
        extra = {'cat_features': list(range(xtrain.shape[1] - 14, xtrain.shape[1]))}
    model = CatBoostClassifier(**BEHAVIOR_CONFIG, **extra)
    model.fit(xtrain, np.asarray([r['y'] for r in training]))
    return predict(model_name, model, evaluation), model


def write_predictions(path: Path, rows: list[dict], probabilities, folds=None):
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError('invalid prediction shape or non-finite values')
    if np.any((values < 0) | (values > 1)):
        raise ValueError('probability outside [0,1]')
    identifiers = [r['ID'] for r in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('duplicate prediction ID')
    columns = ['ID', 'is_bot'] + (['fold'] if folds is not None else [])
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for i, (identifier, probability) in enumerate(zip(identifiers, values)):
            w.writerow([identifier, repr(float(probability))] +
                       ([int(folds[i])] if folds is not None else []))


def provenance() -> dict:
    paths = [ROOT / 'botbench.py', ROOT / 'experiment.py',
             ROOT / 'requirements.txt', ROOT / 'BRIEF.md',
             ROOT / 'evidence/membership-v2.json', ROOT / 'evidence/rules.txt',
             ROOT / 'evidence/representation-audit-v2.md', ROOT / 'evidence/prereg-v2.md',
             ROOT / 'evidence/prereg-v3.md',
             *sorted((ROOT / 'data').glob('*'))]
    return {'sha256': {str(p.relative_to(ROOT)): file_sha(p) for p in paths},
            'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p)
                         for p in ['numpy', 'scipy', 'scikit-learn', 'catboost', 'joblib']}}


def metrics(y, p) -> dict:
    return {'logloss': float(log_loss(y, p, labels=[0, 1])),
            'auc': float(roc_auc_score(y, p))}


def run(model_name: str, out: Path) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    before = provenance()
    for relative in before['sha256']:
        target = out / 'snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    rows = load_dataset(ROOT / 'data', 'train')
    membership = json.loads((ROOT / 'evidence/membership-v2.json').read_text())
    folds = np.asarray(membership_folds(rows, membership))
    predictions = np.full(len(rows), np.nan)
    receipts = []
    for fold in range(5):
        itrain = np.flatnonzero((folds >= 0) & (folds != fold))
        ival = np.flatnonzero(folds == fold)
        training = [rows[i] for i in itrain]
        evaluation = [rows[i] for i in ival]
        training_groups = {r['group'] for r in training}
        if training_groups & {r['group'] for r in evaluation}:
            raise ValueError('train/validation content group leakage')
        begin = time.perf_counter()
        p, model = fit_predict(model_name, training, evaluation)
        predictions[ival] = p
        joblib.dump(model, out / f'fold-{fold}.joblib')
        receipt = {'fold': fold, 'train_rows': len(training),
                   'validation_rows': len(evaluation),
                   'train_IDs': [r['ID'] for r in training],
                   'validation_IDs': [r['ID'] for r in evaluation],
                   'metrics': metrics([r['y'] for r in evaluation], p),
                   'seconds': time.perf_counter() - begin}
        receipts.append(receipt)
        print(json.dumps({k: v for k, v in receipt.items()
                          if not k.endswith('_IDs')}), flush=True)
    idx = np.flatnonzero(folds >= 0)
    dev = [rows[i] for i in idx]
    write_predictions(out / 'oof.csv', dev, predictions[idx], folds[idx])
    if provenance() != before:
        raise ValueError('source/input bytes changed during experiment')
    report = {'schema': 'you_are_bot.diagnostic.v1', 'model': model_name,
              'config': LEXICAL_CONFIG if model_name == 'lexical' else BEHAVIOR_CONFIG,
              'provenance': before,
              'metrics': metrics([r['y'] for r in dev], predictions[idx]),
              'folds': receipts, 'seconds': time.perf_counter() - started,
              'confirmation_opened': False, 'submitted': False,
              'artifacts': {p.name: file_sha(p) for p in sorted(out.glob('fold-*.joblib'))},
              'oof_sha256': file_sha(out / 'oof.csv')}
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ['model', 'metrics', 'seconds',
                                             'confirmation_opened', 'submitted']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['behavior', 'lexical', 'hybrid', 'template'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.out)
