import copy
import importlib
import json
from pathlib import Path


def test_eda_computes_aggregates_without_raw_text_ids_or_test_labels(tiny_data):
    root = Path(__file__).resolve().parents[1]
    assert (root / 'src/eda.py').exists(), 'missing public-safe EDA'
    eda = importlib.import_module('eda')
    _, train, test, membership, _ = tiny_data
    from guards import validate_membership
    # Same-speaker adjacency must be measured from sequence, not alternating indices.
    for row in train[:2]:
        row['messages'][1]['participant_index'] = 0
    result = eda.summarize(train, test, validate_membership(train, membership))
    assert result['train']['dialogs'] == 12
    assert result['test']['participant_rows'] == 4
    assert result['train']['class_counts'] == {'0': 12, '1': 12}
    assert result['train']['pair_types'] == {'human_bot': 12}
    assert result['train']['same_speaker_adjacent_pairs'] == 2
    assert result['train']['missing_fields'] == 0
    assert result['train']['raw_schema']['message_field_sets'] == [['message', 'participant_index', 'text']]
    assert result['train']['raw_schema']['participant_indices'] == [0, 1]
    assert result['train']['raw_schema']['message_numbers_contiguous'] is True
    assert result['train']['leading_whitespace_messages'] == 12
    assert sum(result['train']['message_chars']['histogram']['counts']) == 36
    assert result['membership']['dev_rows'] == 20
    assert result['membership']['confirmation_rows'] == 4
    assert 'class_counts' not in result['test'] and 'pair_types' not in result['test']
    serialized = json.dumps(result)
    assert train[0]['ID'] not in serialized and train[0]['messages'][0]['text'] not in serialized
    with_labels = copy.deepcopy(test)
    for row in with_labels:
        row['y'] = 1
    try:
        eda.summarize(train, with_labels, validate_membership(train, membership))
    except ValueError as exc:
        assert 'test target' in str(exc)
    else:
        raise AssertionError('test targets silently accepted')
