"""Small real stdlib SQLite + JSON tracker, one exclusive directory per run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

import numpy as np

from guards import sha256

CONTROL_FILES = {'tracking.sqlite3', 'tracking.json', 'tracking.json.tmp'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError('metadata must be finite JSON-compatible values')


def dumps(value):
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def no_symlink(path: Path):
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('symlink path refused')


class RunTracker:
    def __init__(self, output_dir: Path, run_id=None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ-') + uuid4().hex[:8]
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', self.run_id):
            raise ValueError('invalid run ID')
        output_dir = Path(os.path.abspath(output_dir))
        no_symlink(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.directory = output_dir / self.run_id
        self.directory.mkdir(exist_ok=False)
        self.connection = sqlite3.connect(self.directory / 'tracking.sqlite3')
        self.connection.executescript('''
            PRAGMA foreign_keys=ON;
            CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_utc TEXT NOT NULL,
                ended_utc TEXT, status TEXT NOT NULL CHECK(status IN ('RUNNING','FINISHED','FAILED')),
                error_json TEXT);
            CREATE TABLE params (run_id TEXT NOT NULL REFERENCES runs, name TEXT NOT NULL,
                value_json TEXT NOT NULL, PRIMARY KEY(run_id,name));
            CREATE TABLE metrics (run_id TEXT NOT NULL REFERENCES runs, name TEXT NOT NULL,
                value REAL NOT NULL, PRIMARY KEY(run_id,name));
            CREATE TABLE artifacts (run_id TEXT NOT NULL REFERENCES runs, path TEXT NOT NULL,
                sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, PRIMARY KEY(run_id,path));
        ''')
        self.status = 'RUNNING'
        self.stage = 'created'
        self.connection.execute('INSERT INTO runs VALUES (?, ?, NULL, ?, NULL)',
                                (self.run_id, utc_now(), self.status))
        self.connection.commit()  # RUNNING receipt survives later transaction rollback.
        self._export()
        self.event('created')

    def _active(self):
        if self.status != 'RUNNING':
            raise RuntimeError('run is terminal and immutable')

    def path(self, relative: str):
        relative_path = Path(relative)
        if relative_path.is_absolute() or not relative_path.parts or '..' in relative_path.parts:
            raise ValueError('artifact path must be run-relative')
        path = self.directory / relative_path
        no_symlink(path)
        if not path.resolve().is_relative_to(self.directory):
            raise ValueError('artifact escapes owned run')
        return path

    def write_json(self, relative, value):
        path = self.path(relative)
        if path.exists():
            raise FileExistsError(relative)
        self._active()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            f.write(dumps(value) + '\n')
        return path

    def event(self, stage, **values):
        self._active()
        self.stage = stage
        path = self.path('run.log.jsonl')
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(jsonable(dict(utc=utc_now(), stage=stage, **values)), ensure_ascii=False) + '\n')

    def params(self, name, value):
        self._active()
        self.connection.execute('INSERT INTO params VALUES (?, ?, ?)', (self.run_id, name, dumps(value)))
        self.connection.commit()
        self._export()

    def metric(self, name, value):
        self._active()
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('metric must be finite')
        self.connection.execute('INSERT INTO metrics VALUES (?, ?, ?)', (self.run_id, name, value))
        self.connection.commit()
        self._export()

    def artifact(self, relative):
        self._active()
        path = self.path(relative)
        if not path.is_file() or relative in CONTROL_FILES:
            raise ValueError('artifact must be an owned ordinary payload file')
        record = (self.run_id, path.relative_to(self.directory).as_posix(), sha256(path), path.stat().st_size)
        self.connection.execute('INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?)', record)
        self.connection.commit()
        return dict(sha256=record[2], bytes=record[3])

    def collect_artifacts(self):
        self._active()
        # Never traverse links, parent evidence, or any directory outside this run.
        for base, directories, files in os.walk(self.directory, followlinks=False):
            for name in directories + files:
                path = Path(base) / name
                no_symlink(path)
            for name in sorted(files):
                path = Path(base) / name
                relative = path.relative_to(self.directory).as_posix()
                if relative not in CONTROL_FILES:
                    self.artifact(relative)
        self._export()

    def _export(self):
        # Fresh SQL readback, not an in-memory success receipt.
        row = self.connection.execute('SELECT * FROM runs WHERE run_id=?', (self.run_id,)).fetchone()
        receipt = dict(schema='coursework.tracking.v1', backend='stdlib SQLite + JSON (not MLflow)',
                       run_id=row[0], started_utc=row[1], ended_utc=row[2], status=row[3],
                       error=json.loads(row[4]) if row[4] else None,
                       params={n: json.loads(v) for n, v in self.connection.execute(
                           'SELECT name,value_json FROM params WHERE run_id=? ORDER BY name', (self.run_id,))},
                       metrics=dict(self.connection.execute('SELECT name,value FROM metrics WHERE run_id=? ORDER BY name', (self.run_id,))),
                       artifacts={p: dict(sha256=h, bytes=s) for p, h, s in self.connection.execute(
                           'SELECT path,sha256,bytes FROM artifacts WHERE run_id=? ORDER BY path', (self.run_id,))},
                       control_files_note='SQLite and tracking.json are receipt envelopes; self-referential hashes excluded.')
        tmp = self.directory / 'tracking.json.tmp'
        no_symlink(tmp)
        with tmp.open('w', encoding='utf-8') as f:
            f.write(dumps(receipt) + '\n')
        os.replace(tmp, self.directory / 'tracking.json')
        return receipt

    def finish(self):
        self._active()
        self.event('pipeline_finished')
        self.collect_artifacts()
        self._terminal('FINISHED', None)

    def _terminal(self, status, error):
        self.connection.rollback()
        self.connection.execute('UPDATE runs SET ended_utc=?,status=?,error_json=? WHERE run_id=?',
                                (utc_now(), status, dumps(error) if error else None, self.run_id))
        self.connection.commit()
        receipt = self._export()
        if receipt['status'] != status:
            raise RuntimeError('tracker terminal readback mismatch')
        # Only terminal after SQL AND JSON projection succeeded. If export
        # raises, __exit__ still records FAILED rather than fake FINISHED.
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if self.status == 'RUNNING':
                # Roll back first: artifact registration must never commit an
                # interrupted metric/parameter transaction accidentally.
                self.connection.rollback()
                failed_stage = self.stage
                error_type = kind.__name__ if kind else 'IncompleteRun'
                try:
                    self.event('failed', error_type=error_type, failed_stage=failed_stage)
                    self.collect_artifacts()
                except (OSError, ValueError, sqlite3.Error):
                    # Even unsafe partial artifacts/disk errors cannot turn a
                    # known failure into FINISHED or erase its SQL receipt.
                    self.connection.rollback()
                self._terminal('FAILED', {'type': error_type, 'stage': failed_stage})
        finally:
            self.connection.close()
        return False
