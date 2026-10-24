"""Aggregate-only EDA. Never export message text, identifiers or test targets."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from uuid import UUID

import numpy as np


def distribution(values, edges):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        raise ValueError('empty EDA distribution')
    counts, _ = np.histogram(values, bins=edges)
    return dict(count=int(values.size), min=float(values.min()), max=float(values.max()),
                mean=float(values.mean()), median=float(np.median(values)),
                quantiles={str(q): float(np.quantile(values, q)) for q in (.1, .25, .75, .9, .99)},
                histogram={'edges': list(edges), 'counts': counts.tolist(),
                           'note': 'Last finite edge is expanded to cover all observed values.'})


def bins_covering(values, base):
    return [*base, max(base[-1] + 1, max(values) + 1)]


def dialog_contents(rows):
    dialogs = {r['dialog_id']: r['messages'] for r in rows}
    return {did: hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True,
             separators=(',', ':')).encode()).hexdigest() for did, messages in dialogs.items()}


def split_summary(rows, labelled):
    dialogs = {r['dialog_id']: r['messages'] for r in rows}
    messages = [m for conversation in dialogs.values() for m in conversation]
    chars = [len(m['text']) for m in messages]
    words = [len(re.findall(r'\w+', m['text'])) for m in messages]
    lengths = [len(conversation) for conversation in dialogs.values()]
    participant_chars = [len(r['own_text']) for r in rows]
    contents = Counter(dialog_contents(rows).values())
    components = Counter(r['group'] for r in rows)
    result: dict = dict(dialogs=len(dialogs), participant_rows=len(rows), messages=len(messages),
                  raw_schema={'message_field_sets': [list(fields) for fields in sorted({tuple(sorted(m)) for m in messages})],
                              'participant_indices': sorted({m['participant_index'] for m in messages}),
                              'message_numbers_contiguous': all(sorted(m['message'] for m in conversation) == list(range(len(conversation))) for conversation in dialogs.values()),
                              'dialog_uuid_version_counts': {str(k): v for k, v in sorted(Counter(UUID(did).version for did in dialogs).items(), key=lambda item: str(item[0]))}},
                  unique_components=len(components),
                  component_participant_rows_histogram={str(k): v for k, v in sorted(Counter(components.values()).items())},
                  unique_exact_dialog_contents=len(contents),
                  exact_duplicate_excess_dialogs=sum(n - 1 for n in contents.values()),
                  exact_duplicate_content_clusters=sum(n > 1 for n in contents.values()),
                  missing_fields=sum(len({'text', 'message', 'participant_index'} - set(m)) for m in messages),
                  empty_messages=sum(not m['text'] for m in messages),
                  whitespace_only_messages=sum(bool(m['text']) and not m['text'].strip() for m in messages),
                  leading_whitespace_messages=sum(bool(m['text']) and m['text'][0].isspace() for m in messages),
                  trailing_whitespace_messages=sum(bool(m['text']) and m['text'][-1].isspace() for m in messages),
                  newline_messages=sum('\n' in m['text'] for m in messages),
                  same_speaker_adjacent_pairs=sum(a['participant_index'] == b['participant_index']
                        for conversation in dialogs.values() for a, b in zip(conversation, conversation[1:])),
                  dialogs_with_same_speaker_adjacency=sum(any(a['participant_index'] == b['participant_index']
                        for a, b in zip(conversation, conversation[1:])) for conversation in dialogs.values()),
                  uuid_mentions=sum(bool(re.search(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b', m['text'])) for m in messages),
                  serialized_dialog_reference_messages=sum(bool(re.search(r'dialog_id\s*=\s*UUID\(', m['text'])) for m in messages),
                  message_chars=distribution(chars, bins_covering(chars, [0, 1, 5, 20, 50, 100, 200, 500, 1000, 5000])),
                  message_words=distribution(words, bins_covering(words, [0, 1, 5, 20, 50, 100, 500])),
                  dialog_messages=distribution(lengths, bins_covering(lengths, [0, 2, 5, 10, 20, 50, 100])),
                  participant_chars=distribution(participant_chars, bins_covering(participant_chars, [0, 1, 50, 200, 1000, 5000, 12000])))
    if labelled:
        result['class_counts'] = {str(k): v for k, v in sorted(Counter(r['y'] for r in rows).items())}
        pairs = {}
        for r in rows:
            pairs.setdefault(r['dialog_id'], {})[r['participant_index']] = r['y']
        names = {0: 'human_human', 1: 'human_bot', 2: 'bot_bot'}
        result['pair_types'] = dict(sorted(Counter(names[sum(p.values())] for p in pairs.values()).items()))
    return result


def summarize(train, test, folds):
    if any('y' in r or 'is_bot' in r for r in test):
        raise ValueError('unexpected test target in EDA')
    left, right = split_summary(train, True), split_summary(test, False)
    return dict(schema='coursework.eda.v1', public_safe=True,
                note='Only aggregates. Official test labels unavailable; no test class/pair estimates.',
                train=left, test=right,
                train_vs_test={'exact_content_overlap_dialogs': len(set(dialog_contents(train).values()) & set(dialog_contents(test).values())),
                               'message_length_mean_delta_test_minus_train': right['message_chars']['mean'] - left['message_chars']['mean'],
                               'dialog_length_mean_delta_test_minus_train': right['dialog_messages']['mean'] - left['dialog_messages']['mean']},
                membership={'dev_rows': int(np.sum(folds >= 0)), 'confirmation_rows': int(np.sum(folds == -1)),
                            'fold_rows': {str(k): int(np.sum(folds == k)) for k in range(5)},
                            'grouping': 'Exact dialog content + explicit serialized UUID references; connected components never cross CV folds.'})
