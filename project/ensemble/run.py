#!/usr/bin/env python3
"""Reconstruct submission #10588 from genuine saved CatBoost + LLM predictions.

This is deterministic ensemble replay, NOT fresh training or fresh LLM inference.
Uses only Python's standard library. Never calls a network or submission API.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED_ROWS = 758
COMPONENTS_SHA256 = 'ed36be23cfb492ac2dfdd77298bf998b64c900703db3b06f97af2a4e4bb6ab6e'
SUBMISSION_SHA256 = '4bde02a00f00d009ad69c3aa967c00e468a41c0bcb2b90f65b93416abb3d4639'
ID_PATTERN = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}_[01]')


def reconstruct():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'you-are-bot.ensemble-replay.v1'
            or manifest.get('mode') != 'saved_component_predictions_replay'
            or manifest.get('rows') != EXPECTED_ROWS
            or manifest.get('weights') != {'catboost': 0.5, 'llm': 0.5}
            or manifest.get('components_sha256') != COMPONENTS_SHA256
            or manifest.get('submission_sha256') != SUBMISSION_SHA256
            or manifest.get('historical_submission_id') != 10588):
        raise ValueError('Frozen ensemble manifest differs from the accepted recipe')
    data = (ROOT / 'components.csv').read_bytes()
    if hashlib.sha256(data).hexdigest() != COMPONENTS_SHA256:
        raise ValueError('components.csv SHA-256 mismatch; output not created')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    if reader.fieldnames != ['ID', 'catboost', 'llm']:
        raise ValueError('Invalid component schema')
    rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError('Invalid participant count')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow(['ID', 'is_bot'])
    seen = set()
    for row in rows:
        if set(row) != {'ID', 'catboost', 'llm'}:
            raise ValueError('Invalid component row')
        identifier = row['ID']
        if not ID_PATTERN.fullmatch(identifier) or identifier in seen:
            raise ValueError('Invalid or duplicate participant ID')
        seen.add(identifier)
        native, llm = float(row['catboost']), float(row['llm'])
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in (native, llm)):
            raise ValueError('Component probability must be finite and in [0,1]')
        # Same two scalar multiplications and addition as the historical recipe.
        writer.writerow([identifier, 0.5 * native + 0.5 * llm])
    payload = stream.getvalue().encode('utf-8')
    if hashlib.sha256(payload).hexdigest() != SUBMISSION_SHA256:
        raise ValueError('Reconstructed bytes differ from official submission #10588')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('submission.csv'),
                        help='New output CSV; existing files are never overwritten')
    args = parser.parse_args()
    try:
        payload = reconstruct()
        with args.output.open('xb') as file:
            file.write(payload)
        if hashlib.sha256(args.output.read_bytes()).hexdigest() != SUBMISSION_SHA256:
            raise ValueError('Written file failed SHA-256 readback')
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps({'status': 'VERIFIED', 'mode': 'saved_component_predictions_replay',
                      'rows': EXPECTED_ROWS, 'sha256': SUBMISSION_SHA256,
                      'output': str(args.output.resolve()), 'historical_submission_id': 10588,
                      'historical_public_logloss': '0.1303426569',
                      'score_recomputed_locally': False, 'submitted': False}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
