import importlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest


def tracker_class():
    assert (Path(__file__).resolve().parents[1] / 'src/tracking.py').exists(), 'missing real tracker'
    return importlib.import_module('tracking').RunTracker


def test_failed_run_receipt_survives_exception_and_sql_rollback(tmp_path):
    Tracker = tracker_class()
    with pytest.raises(RuntimeError):
        with Tracker(tmp_path, 'failure-test') as run:
            run.params('fixed', {'seed': np.int64(1729), 'verbose': False})
            run.write_json('partial.json', {'unit_test_artifact': True})
            run.connection.execute("INSERT INTO metrics VALUES (?, ?, ?)", ('failure-test', 'uncommitted', 42.0))
            raise RuntimeError('private raw text must not appear in public receipt')
    directory = tmp_path / 'failure-test'
    with sqlite3.connect(directory / 'tracking.sqlite3') as db:
        assert db.execute('SELECT status FROM runs WHERE run_id=?', ('failure-test',)).fetchone() == ('FAILED',)
        assert db.execute('SELECT value FROM metrics WHERE name=?', ('uncommitted',)).fetchone() is None
    receipt = json.loads((directory / 'tracking.json').read_text())
    assert receipt['status'] == 'FAILED'
    assert receipt['error']['type'] == 'RuntimeError'
    assert 'partial.json' in receipt['artifacts']
    assert 'private raw text' not in (directory / 'tracking.json').read_text()
    with pytest.raises(FileExistsError):
        Tracker(tmp_path, 'failure-test')


def test_params_metrics_artifact_hashes_readback_and_finished_immutability(tmp_path):
    Tracker = tracker_class()
    with Tracker(tmp_path, 'complete') as run:
        run.params('effective/fold-0', {'depth': 5, 'learning_rate': np.float64(.04), 'all_defaults': {'a': [1, 2]}})
        run.metric('oof/logloss', np.float64(.2))
        run.write_json('report.json', {'actual': True})
        run.artifact('report.json')
        run.finish()
    receipt = json.loads((run.directory / 'tracking.json').read_text())
    assert receipt['status'] == 'FINISHED'
    assert receipt['params']['effective/fold-0']['all_defaults']['a'] == [1, 2]
    assert receipt['metrics']['oof/logloss'] == .2
    assert receipt['artifacts']['report.json']['bytes'] > 0
    assert len(receipt['artifacts']['report.json']['sha256']) == 64
    assert receipt['started_utc'].endswith('+00:00') and receipt['ended_utc'].endswith('+00:00')
    with pytest.raises(RuntimeError, match='terminal'):
        run.params('after', {})
    with pytest.raises(FileExistsError):
        run.write_json('report.json', {})


def test_terminal_json_export_error_is_failed_not_fake_finished(tmp_path, monkeypatch):
    Tracker = tracker_class()
    with pytest.raises(OSError, match='unit export failure'):
        with Tracker(tmp_path, 'export-failure') as run:
            real_export = run._export
            raised = False
            def failing_export_once():
                nonlocal raised
                status = run.connection.execute('SELECT status FROM runs WHERE run_id=?', (run.run_id,)).fetchone()[0]
                if status == 'FINISHED' and not raised:
                    raised = True
                    raise OSError('unit export failure')
                return real_export()
            monkeypatch.setattr(run, '_export', failing_export_once)
            run.finish()
    with sqlite3.connect(tmp_path / 'export-failure/tracking.sqlite3') as db:
        assert db.execute('SELECT status FROM runs').fetchone() == ('FAILED',)
    assert json.loads((tmp_path / 'export-failure/tracking.json').read_text())['status'] == 'FAILED'


def test_path_confinement_symlink_and_run_id_validation(tmp_path):
    Tracker = tracker_class()
    for identifier in ('../escape', '/absolute', '.', 'two/levels', 'x' * 100):
        with pytest.raises(ValueError):
            Tracker(tmp_path, identifier)
    with Tracker(tmp_path, 'confined') as run:
        foreign = tmp_path / 'foreign'; foreign.write_text('foreign data')
        (run.directory / 'link').symlink_to(foreign)
        with pytest.raises(ValueError, match='symlink'):
            run.artifact('link')
        with pytest.raises(ValueError):
            run.artifact('../foreign')
        with pytest.raises(ValueError):
            run.collect_artifacts()
