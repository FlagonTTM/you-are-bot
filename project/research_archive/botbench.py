"""Official-data loading and target-free representations for You Are Bot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_probabilities(path: Path, *, has_fold: bool = False) -> dict[str, float]:
    columns = ['ID', 'is_bot'] + (['fold'] if has_fold else [])
    result = {}
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError('probability CSV header mismatch')
        for row in reader:
            if set(row) != set(columns) or not row['ID'] or row['is_bot'] is None:
                raise ValueError('probability CSV row schema mismatch')
            if row['ID'] in result:
                raise ValueError('duplicate probability ID')
            value = float(row['is_bot'])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid probability value')
            result[row['ID']] = value
    return result


def load_dataset(root: Path, split: str) -> list[dict]:
    if split not in {'train', 'test'}:
        raise ValueError('split must be train or test')
    dialogs = json.loads((root / f'{split}.json').read_text(encoding='utf-8'))
    with (root / f'y{split}.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns or not {'dialog_id', 'participant_index'} <= set(columns):
            raise ValueError('missing participant keys')
        if split == 'test' and 'is_bot' in columns:
            raise ValueError('unexpected test target; refuse to read labels')
        if split == 'train' and 'is_bot' not in columns:
            raise ValueError('missing training target')
        labels = list(reader)
    validated = {}
    for did, messages in dialogs.items():
        if not isinstance(messages, list) or not messages:
            raise ValueError(f'empty dialog: {did}')
        numbers = []
        for m in messages:
            if (set(m) != {'text', 'message', 'participant_index'}
                    or not isinstance(m['text'], str)
                    or type(m['message']) is not int
                    or type(m['participant_index']) is not int
                    or m['participant_index'] not in (0, 1)):
                raise ValueError(f'invalid message in {did}')
            numbers.append(m['message'])
        if sorted(numbers) != list(range(len(messages))):
            raise ValueError(f'message sequence is not contiguous: {did}')
        ordered = sorted(messages, key=lambda m: m['message'])
        canonical = json.dumps(ordered, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'))
        validated[did] = (ordered, hashlib.sha256(canonical.encode()).hexdigest())
    rows, seen = [], set()
    for label in labels:
        did, participant = label['dialog_id'], int(label['participant_index'])
        identifier = f'{did}_{participant}'
        if participant not in (0, 1) or identifier in seen:
            raise ValueError('invalid or duplicate participant key')
        if split == 'test' and label.get('ID') != identifier:
            raise ValueError('official ID differs from participant key')
        seen.add(identifier)
        messages, group = validated[did]
        own = [m['text'] for m in messages if m['participant_index'] == participant]
        other = [m['text'] for m in messages if m['participant_index'] != participant]
        row = {'ID': identifier, 'dialog_id': did, 'participant_index': participant,
               'group': group, 'messages': messages, 'own_text': '\n'.join(own),
               'other_text': '\n'.join(other)}
        if split == 'train':
            row['y'] = int(label['is_bot'])
            if row['y'] not in (0, 1):
                raise ValueError('training target is not binary')
        rows.append(row)
    expected = {f'{did}_{p}' for did in dialogs for p in (0, 1)}
    if seen != expected:
        raise ValueError('participant membership does not cover all dialogs')
    return link_groups(rows)


def link_groups(rows: list[dict]) -> list[dict]:
    """Union exact dialog duplicates and explicit serialized dialog references."""
    parents = {r['dialog_id']: r['dialog_id'] for r in rows}

    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    contents, visited = {}, set()
    for r in rows:
        did = r['dialog_id']
        if did in visited:
            continue
        visited.add(did)
        if r['group'] in contents:
            union(did, contents[r['group']])
        else:
            contents[r['group']] = did
        for m in r['messages']:
            for ref in re.findall(r'dialog_id\s*=\s*UUID\([\'\"]([a-fA-F0-9-]+)[\'\"]\)', m['text']):
                if ref in parents:
                    union(did, ref)
    return [{**r, 'group': find(r['dialog_id'])} for r in rows]


def participant_features(messages: list[dict], participant: int) -> dict[str, float]:
    """Pure target-free signals; message order, not alternating-row assumption."""
    own = [m['text'] for m in messages if m['participant_index'] == participant]
    text = '\n'.join(own)
    n, chars = len(own), max(1, len(text))
    words = re.findall(r'\w+', text.lower())
    normalize = lambda s: ' '.join(s.casefold().split())
    lengths = [len(t) for t in own] or [0]
    f = {'n_messages': float(n), 'chars': float(len(text)),
         'words': float(len(words)), 'unique_words': float(len(set(words))),
         'word_diversity': len(set(words)) / max(1, len(words)),
         'unique_messages': len(set(own)) / max(1, n),
         'participant_index': float(participant),
         'starts_dialog': float(messages[0]['participant_index'] == participant),
         'ends_dialog': float(messages[-1]['participant_index'] == participant),
         'dialog_messages': float(len(messages)),
         'len_mean': float(np.mean(lengths)), 'len_std': float(np.std(lengths)),
         'len_min': float(min(lengths)), 'len_max': float(max(lengths)),
         'len_median': float(np.median(lengths)),
         'word_len_mean': float(np.mean([len(w) for w in words] or [0])),
         'longest_word': float(max([len(w) for w in words] or [0]))}
    char_counts = Counter(text)
    f['char_entropy'] = -sum((c / chars) * math.log(c / chars) for c in char_counts.values())
    for name, predicate in {
        'upper': str.isupper, 'lower': str.islower, 'digits': str.isdigit,
        'spaces': str.isspace, 'cyrillic': lambda c: '\u0400' <= c <= '\u04ff',
        'latin': lambda c: 'a' <= c.lower() <= 'z',
        'nonword': lambda c: not (c.isalnum() or c.isspace()),
    }.items():
        f[name + '_ratio'] = sum(predicate(c) for c in text) / chars
    for symbol, name in [('?', 'question'), ('!', 'exclamation'), ('.', 'period'),
                         (',', 'comma'), ('\n', 'newline'), (':', 'colon'),
                         ('(', 'left_paren'), (')', 'right_paren'), ('*', 'star'),
                         ('"', 'quote'), ('—', 'em_dash')]:
        f[name + '_ratio'] = text.count(symbol) / chars
    for name, predicate in {
        'short': lambda t: len(t) < 5,
        'long': lambda t: len(t) > 200,
        'start_upper': lambda t: bool(t) and t[0].isupper(),
        'end_period': lambda t: t.endswith('.'),
        'end_paren': lambda t: t.endswith(')'),
        'contains_url': lambda t: 'http' in t.lower(),
        'contains_newline': lambda t: '\n' in t,
    }.items():
        f[name + '_fraction'] = sum(predicate(t) for t in own) / max(1, n)
    replies, exact, normalized, similarities, length_ratios, same_turns = 0, 0, 0, [], [], 0
    last_other = None
    prev_participant = None
    for m in messages:
        p, t = m['participant_index'], m['text']
        if p == participant:
            same_turns += prev_participant == participant
            if last_other is not None:
                replies += 1
                exact += t == last_other
                normalized += normalize(t) == normalize(last_other)
                similarities.append(SequenceMatcher(None, normalize(t)[:1500],
                                                     normalize(last_other)[:1500],
                                                     autojunk=False).ratio())
                length_ratios.append(len(t) / max(1, len(last_other)))
        else:
            last_other = t
        prev_participant = p
    f.update({'reply_count': float(replies), 'echo_exact': exact / max(1, replies),
              'echo_normalized': normalized / max(1, replies),
              'echo_similarity_mean': float(np.mean(similarities or [0])),
              'echo_similarity_max': float(max(similarities or [0])),
              'reply_length_ratio': float(np.mean(length_ratios or [0])),
              'same_participant_turns': float(same_turns)})
    return f


def numeric_features(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    cache, features = {}, []
    for r in rows:
        key = r['dialog_id']
        if key not in cache:
            cache[key] = [participant_features(r['messages'], p) for p in (0, 1)]
        own = cache[key][r['participant_index']]
        other = cache[key][1 - r['participant_index']]
        f = {f'own_{k}': v for k, v in own.items()}
        f.update({f'other_{k}': v for k, v in other.items()})
        f.update({f'difference_{k}': v - other[k] for k, v in own.items()})
        features.append(f)
    names = sorted(features[0])
    matrix = np.asarray([[f[k] for k in names] for f in features], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('non-finite numeric representation')
    return matrix, names


def membership_folds(rows: list[dict], membership: list[dict]) -> list[int]:
    ids = [r['ID'] for r in rows]
    if ids != [m['ID'] for m in membership] or len(ids) != len(set(ids)):
        raise ValueError('membership ID/order/cardinality mismatch')
    assigned, result = {}, []
    for r, m in zip(rows, membership):
        if m['group'] != r['group'] or type(m['fold']) is not int or m['fold'] not in range(-1, 5):
            raise ValueError('membership group/fold contract mismatch')
        g, f = m['group'], m['fold']
        if g in assigned and assigned[g] != f:
            raise ValueError('related dialog component spans multiple folds')
        assigned[g] = f
        result.append(f)
    return result


def make_membership(rows: list[dict]) -> list[int]:
    """-1 is sealed confirmation; 0..4 are development OOF folds."""
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    groups = np.asarray([r['group'] for r in rows])
    y = np.asarray([r['y'] for r in rows])
    dev, confirmation = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=1729).split(y, y, groups))
    membership = np.full(len(rows), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1729)
    for fold, (training, validation) in enumerate(splitter.split(dev, y[dev], groups[dev])):
        assert not set(groups[dev[training]]) & set(groups[dev[validation]])
        membership[dev[validation]] = fold
    assert not set(groups[dev]) & set(groups[confirmation])
    return membership.tolist()
