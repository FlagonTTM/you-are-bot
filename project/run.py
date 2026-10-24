#!/usr/bin/env python3
"""One official-data entry; no model/config overrides or external code imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Offline frozen native CatBoost: EDA, grouped 5-fold CV, full refit, CSV, replay, SQLite/JSON.')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory with the five unchanged official input files')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs', help='Parent directory for a NEW immutable run namespace')
    parser.add_argument('--run-id', default=None, help='Unique run name; UTC timestamp + random suffix by default')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    try:
        from pipeline import run
        run(args.data_dir, args.output_dir, args.run_id)
    except (Exception, KeyboardInterrupt) as exc:
        # Detailed type+stage are in the FAILED receipt; do not export private
        # absolute paths, arbitrary raw rows, credentials or tracebacks.
        print(json.dumps({'status': 'FAILED', 'error_type': type(exc).__name__,
                          'hint': 'Read tracking.json error.type/error.stage in the requested run namespace; check input SHA/schema, lock and permissions.'}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
