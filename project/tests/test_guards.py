import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def guards():
    assert (Path(__file__).resolve().parents[1] / 'src/guards.py').exists(), 'missing preflight guards'
    return importlib.import_module('guards')


def test_five_inputs_hashes_sample_order_and_target_rejection(tiny_data):
    g = guards()
    data, train, test, membership, hashes = tiny_data
    loaded_train, loaded_test, sample, before = g.validate_data(data, hashes)
    assert [r['ID'] for r in loaded_train] == [r['ID'] for r in train]
    assert sample == [r['ID'] for r in reversed(test)]
    assert before == hashes
    assert 'y' not in loaded_test[0]
    # An explicit label header is rejected even if its bytes were authorized.
    p = data / 'ytest.csv'
    p.write_text(p.read_text().replace('ID', 'is_bot'))
    with pytest.raises(ValueError, match='test target'):
        g.validate_data(data, hashes)


def test_membership_binding_components_and_cv_fit_exclusion(tiny_data):
    g = guards()
    _, train, _, membership, _ = tiny_data
    folds = g.validate_membership(train, membership)
    splits = list(g.cv_splits(train, folds))
    assert len(splits) == 5
    for fold, itrain, ival in splits:
        assert all(folds[i] >= 0 and folds[i] != fold for i in itrain)
        assert all(folds[i] == fold for i in ival)
        assert not {train[i]['group'] for i in itrain} & {train[i]['group'] for i in ival}
    bad = copy.deepcopy(membership); bad[0]['dialog_id'] = 'foreign'
    with pytest.raises(ValueError, match='dialog'):
        g.validate_membership(train, bad)
    bad = copy.deepcopy(membership); bad[0]['fold'] = 0
    with pytest.raises(ValueError, match='component'):
        g.validate_membership(train, bad)
    with pytest.raises(ValueError, match='ID'):
        g.validate_membership(train, list(reversed(membership)))


def test_source_closure_and_immutability(tmp_path):
    g = guards()
    (tmp_path / 'src').mkdir()
    p = tmp_path / 'src/a.py'; p.write_text('value = 1\n')
    before = g.source_manifest(tmp_path)
    assert before.keys() == {'src/a.py'}
    (tmp_path / 'presentation.html').write_text('mutable parent deck')
    (tmp_path / 'evidence').mkdir()
    (tmp_path / 'evidence/history.json').write_text('mutable archive')
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/parent.py').write_text('mutable parent tool')
    assert before == g.source_manifest(tmp_path)
    p.write_text('value = 2\n')
    with pytest.raises(ValueError, match='changed'):
        g.assert_unchanged(before, g.source_manifest(tmp_path), 'source')
    foreign = tmp_path / 'private'; foreign.write_text('not part of source closure')
    (tmp_path / 'src/link.py').symlink_to(foreign)
    with pytest.raises(ValueError, match='symlink'):
        g.source_manifest(tmp_path)


def test_features_are_target_and_identity_free_full_bank(tiny_data):
    g = guards()
    from experiment import model_input
    _, train, _, _, _ = tiny_data
    blind = g.blind_rows(train)
    assert all('y' not in r for r in blind)
    expected = model_input(train, 'template')
    altered = copy.deepcopy(train)
    for r in altered:
        r.update(y=1-r['y'], ID='foreign-id', dialog_id='changed-' + r['dialog_id'], group='foreign')
    np.testing.assert_array_equal(expected, model_input(altered, 'template'))
    np.testing.assert_array_equal(expected, model_input(blind, 'template'))
    assert expected.shape == (len(train), 176)


def test_hash_missing_duplicate_json_and_sample_ids_fail_closed(tiny_data):
    g = guards()
    data, _, _, _, hashes = tiny_data
    (data / 'train.json').write_text('{"x":[],"x":[]}')
    with pytest.raises(ValueError, match='SHA'):
        g.validate_data(data, hashes)
    with pytest.raises(ValueError, match='duplicate'):
        g.strict_json(data / 'train.json')
    (data / 'test.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        g.validate_data(data, hashes)
