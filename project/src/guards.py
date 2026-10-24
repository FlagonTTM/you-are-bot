"""Fail-closed official input and immutable scientific-closure checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from botbench import load_dataset, membership_folds

REQUIRED = ('train.json', 'ytrain.csv', 'test.json', 'ytest.csv', 'sample_submission.csv')
OFFICIAL_HASHES = {
    'sample_submission.csv': 'bf0ecf417e9ed88001370c437949c7540c09449c0b3e5f9746627c67ec1b503f',
    'test.json': 'a330a9e00e724cb6fe3541d89b9646ab2aef83c025f0b0bcc24811ceddedf428',
    'train.json': 'ca3795aa47b8791174a9225a1fe3219684dc388c88922053afdab411bf44ea34',
    'ytest.csv': '46463c8de28dbbeaa3b0e763c62c1a4a8b251e9df9c79c9e64a559224fe8e7aa',
    'ytrain.csv': '7977ce4773cade43f27459711b2ca943addf284cd6c0ea9b830c71ebc7d5b233',
}
MEMBERSHIP_SHA = 'e553e7bcdfa98b398b0a534a5d9fc095c5a19f60054f87c365cbe705d6b04dee'
OFFICIAL_COUNTS = dict(train_dialogs=1516, test_dialogs=379, train_rows=3032,
                       test_rows=758, dev_rows=2416, confirmation_rows=616)
UUID_PATTERN = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def csv_rows(path: Path, columns: list[str]):
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f'CSV schema mismatch: {path.name}')
        rows = list(reader)
    if any(set(row) != set(columns) or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'CSV row schema mismatch: {path.name}')
    return rows


def data_manifest(data_dir: Path):
    result = {}
    for name in REQUIRED:
        path = data_dir / name
        if path.is_symlink():
            raise ValueError(f'input symlink refused: {name}')
        if not path.is_file():
            raise ValueError(f'missing required input: {name}')
        result[name] = sha256(path)
    return result


def validate_data(data_dir: Path, expected_hashes=None):
    """The CLI always uses OFFICIAL_HASHES. Injection is for tiny unit fixtures."""
    before = data_manifest(data_dir)
    # Header-only target rejection precedes any loading of test values.
    with (data_dir / 'ytest.csv').open(encoding='utf-8', newline='') as f:
        if 'is_bot' in (next(csv.reader(f), []) or []):
            raise ValueError('unexpected test target; refuse to read labels')
    if before != (OFFICIAL_HASHES if expected_hashes is None else expected_hashes):
        raise ValueError('official input SHA-256 mismatch')
    for split in ('train', 'test'):
        raw = strict_json(data_dir / f'{split}.json')
        if not isinstance(raw, dict) or not raw:
            raise ValueError('raw JSON must be a nonempty dialog object')
        if any(not UUID_PATTERN.fullmatch(did) for did in raw):
            raise ValueError('invalid dialog UUID')
        columns = ['dialog_id', 'participant_index', 'is_bot' if split == 'train' else 'ID']
        labels = csv_rows(data_dir / f'y{split}.csv', columns)
        for row in labels:
            if row['participant_index'] not in {'0', '1'}:
                raise ValueError('participant index must be 0/1')
            if split == 'train' and row['is_bot'] not in {'0', '1'}:
                raise ValueError('training target must be 0/1')
    train, test = load_dataset(data_dir, 'train'), load_dataset(data_dir, 'test')
    if {r['dialog_id'] for r in train} & {r['dialog_id'] for r in test}:
        raise ValueError('train/test dialog identifiers overlap')
    sample_rows = csv_rows(data_dir / 'sample_submission.csv', ['ID', 'is_bot'])
    sample = [r['ID'] for r in sample_rows]
    if len(sample) != len(set(sample)) or set(sample) != {r['ID'] for r in test}:
        raise ValueError('sample ID/cardinality mismatch')
    assert_unchanged(before, data_manifest(data_dir), 'inputs')
    return train, test, sample, before


def validate_membership(rows, membership):
    if len(rows) != len(membership):
        raise ValueError('membership ID/cardinality mismatch')
    folds = np.asarray(membership_folds(rows, membership), dtype=int)
    for row, member in zip(rows, membership):
        if member.get('dialog_id') != row['dialog_id']:
            raise ValueError('membership dialog binding mismatch')
    return folds


def validate_counts(train, test, folds, expected=None):
    actual = dict(train_dialogs=len({r['dialog_id'] for r in train}),
                  test_dialogs=len({r['dialog_id'] for r in test}),
                  train_rows=len(train), test_rows=len(test),
                  dev_rows=int(np.sum(folds >= 0)), confirmation_rows=int(np.sum(folds == -1)))
    if actual != (OFFICIAL_COUNTS if expected is None else expected):
        raise ValueError('official dataset cardinality mismatch')
    return actual


def cv_splits(rows, folds):
    folds = np.asarray(folds)
    if folds.shape != (len(rows),) or set(folds.tolist()) != {-1, 0, 1, 2, 3, 4}:
        raise ValueError('expected confirmation plus five nonempty folds')
    assigned = {}
    for row, fold in zip(rows, folds):
        if row['group'] in assigned and assigned[row['group']] != fold:
            raise ValueError('component leakage across folds')
        assigned[row['group']] = fold
    for fold in range(5):
        itrain = np.flatnonzero((folds >= 0) & (folds != fold))
        ival = np.flatnonzero(folds == fold)
        if {rows[i]['group'] for i in itrain} & {rows[i]['group'] for i in ival}:
            raise ValueError('train/validation component leakage')
        yield fold, itrain, ival


def blind_rows(rows):
    return [{k: v for k, v in row.items() if k != 'y'} for row in rows]


def source_manifest(root: Path):
    """Only consumed runtime/config/lock files, never mutable decks or evidence."""
    paths = [*sorted((root / 'src').glob('*.py')),
             *[root / p for p in ('src/origins.json', 'run.py', 'run.sh',
                                  'configs/template.json', 'configs/membership-v2.json',
                                  'requirements.in', 'requirements.lock')]]
    result = {}
    for path in paths:
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
            raise ValueError('source symlink refused')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def assert_unchanged(before, after, scope):
    if before != after:
        raise ValueError(f'{scope} bytes changed during run')
