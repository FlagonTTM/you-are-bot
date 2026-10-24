"""Only tiny native CatBoost fits. No official-data fits in this test suite."""
import importlib
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pipeline_module():
    assert (ROOT / 'src/pipeline.py').is_file(), 'missing end-to-end pipeline'
    return importlib.import_module('pipeline')


def test_tiny_native_vertical_pipeline_cv_refit_replay_tracking(tiny_data, tmp_path, monkeypatch):
    pipeline = pipeline_module()
    import experiment
    from guards import validate_membership
    from tracking import RunTracker
    _, train, test, membership, _ = tiny_data
    folds = validate_membership(train, membership)
    # Unit-only resource reduction; default CLI recipe is independently locked.
    monkeypatch.setattr(experiment, 'BEHAVIOR_CONFIG', dict(experiment.BEHAVIOR_CONFIG, iterations=3, depth=2))
    original_fit = experiment.fit_predict
    calls = []
    def audited_fit(model_name, training, evaluation):
        assert all('y' in r for r in training)
        assert all('y' not in r for r in evaluation)
        calls.append(({r['ID'] for r in training}, {r['ID'] for r in evaluation}))
        return original_fit(model_name, training, evaluation)
    monkeypatch.setattr(experiment, 'fit_predict', audited_fit)
    sample = [r['ID'] for r in reversed(test)]
    with RunTracker(tmp_path / 'runs', 'tiny-unit-not-official') as run:
        report = pipeline.execute_pipeline(run, train, test, folds, sample)
        run.finish()
    assert len(calls) == 6
    confirmation = {r['ID'] for r, fold in zip(train, folds) if fold == -1}
    assert all(not confirmation & training for training, _ in calls[:5])
    assert calls[5][0] == {r['ID'] for r in train}
    assert report['counts']['oof_rows'] == 20 and report['counts']['test_rows'] == 4
    assert report['checkpoint_replay']['all_exact'] is True
    assert report['submitted'] is False and report['http_service_implemented'] is False
    assert report['feature_counts'] == {'numeric': 162, 'categorical': 14, 'total': 176}
    fold_receipts = json.loads((run.directory / 'fold-receipts.json').read_text())
    assert len(fold_receipts) == 5
    for receipt in fold_receipts:
        assert not set(receipt['train_IDs']) & set(receipt['validation_IDs'])
        assert not set(receipt['train_groups']) & set(receipt['validation_groups'])
        assert receipt['component_overlap'] == 0 and receipt['confirmation_rows_in_fit'] == 0
    saved_ids, values = pipeline.read_submission(run.directory / 'native-test.csv', sample)
    assert saved_ids == sample and np.isfinite(values).all()
    receipt = json.loads((run.directory / 'tracking.json').read_text())
    assert receipt['status'] == 'FINISHED'
    assert len([n for n in receipt['params'] if n.startswith('model/')]) == 6
    effective = receipt['params']['model/fold-0']['get_all_params']
    assert effective['iterations'] == 3 and effective['depth'] == 2
    assert 'bootstrap_type' in effective
    assert 'models/full.cbm' in receipt['artifacts']
    assert 'index.html' in receipt['artifacts']
    with sqlite3.connect(run.directory / 'tracking.sqlite3') as db:
        assert db.execute('SELECT value FROM metrics WHERE run_id=? AND name=?',
                          (run.run_id, 'oof/logloss')).fetchone()[0] == report['metrics']['logloss']
    for artifact, record in receipt['artifacts'].items():
        from guards import sha256
        assert sha256(run.directory / artifact) == record['sha256']
        assert (run.directory / artifact).stat().st_size == record['bytes']


def test_checkpoint_helper_strict_replay_and_submission_guards(tiny_data, tmp_path):
    pipeline = pipeline_module()
    from catboost import CatBoostClassifier
    import experiment
    from guards import blind_rows
    _, train, test, _, _ = tiny_data
    x = experiment.model_input(train, 'template')
    model = CatBoostClassifier(iterations=2, depth=2, random_seed=1729,
        cat_features=list(range(162, 176)), thread_count=4, verbose=False, allow_writing_files=False)
    model.fit(x, [r['y'] for r in train])
    blind = blind_rows(test)
    probabilities = experiment.predict('template', model, blind)
    path = tmp_path / 'own.cbm'
    pipeline.save_checkpoint(model, path)
    assert pipeline.replay_checkpoint(path, blind, probabilities)['exact']
    with pytest.raises(ValueError, match='replay'):
        pipeline.replay_checkpoint(path, blind, probabilities + .01)
    with pytest.raises(FileExistsError):
        pipeline.save_checkpoint(model, path)
    csv_path = tmp_path / 'test.csv'
    experiment.write_predictions(csv_path, test, probabilities)
    with pytest.raises(ValueError, match='ID'):
        pipeline.read_submission(csv_path, [r['ID'] for r in reversed(test)])
    csv_path.write_text('ID,is_bot\nx,nan\n')
    with pytest.raises(ValueError, match='probability'):
        pipeline.read_submission(csv_path, ['x'])


def test_cli_one_entry_no_old_root_imports_failed_preflight_and_no_overwrite(tmp_path):
    pipeline_module()
    assert (ROOT / 'run.sh').is_file() and (ROOT / 'run.py').is_file()
    help_result = subprocess.run([str(ROOT / 'run.sh'), '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    for flag in ['--data-dir', '--output-dir', '--run-id']:
        assert flag in help_result.stdout
    assert '--model' not in help_result.stdout
    command = [str(ROOT / 'run.sh'), '--data-dir', str(tmp_path / 'missing'),
               '--output-dir', str(tmp_path / 'runs'), '--run-id', 'preflight-failure']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(result.stderr)['status'] == 'FAILED'
    assert str(ROOT) not in result.stderr and 'Traceback' not in result.stderr
    receipt_path = tmp_path / 'runs/preflight-failure/tracking.json'
    assert json.loads(receipt_path.read_text())['status'] == 'FAILED'
    assert not list((tmp_path / 'runs').rglob('*.cbm'))
    before = receipt_path.read_bytes()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and receipt_path.read_bytes() == before
    import botbench, experiment
    assert Path(botbench.__file__).parent == ROOT / 'src'
    assert Path(experiment.__file__).parent == ROOT / 'src'
