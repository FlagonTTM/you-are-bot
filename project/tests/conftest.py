import csv
import hashlib
import json
from pathlib import Path
import sys
from uuid import UUID

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


@pytest.fixture
def tiny_data(tmp_path):
    """Artificial 12+2 dialogs. Not a competition-data fit or quality estimate."""
    from botbench import load_dataset
    data = tmp_path / 'tiny-data'
    data.mkdir()
    for split, start, size in [('train', 1, 12), ('test', 101, 2)]:
        dialogs, labels = {}, []
        for n in range(start, start + size):
            did = str(UUID(int=n))
            dialogs[did] = [
                {'text': f'Обычный ответ номер {n}', 'message': 0, 'participant_index': 0},
                {'text': f'  Structured reply {n}!\n', 'message': 1, 'participant_index': 1},
                {'text': f'Ещё ответ {n}', 'message': 2, 'participant_index': 0},
            ]
            for p in (0, 1):
                row = {'dialog_id': did, 'participant_index': p}
                row.update({'is_bot': p} if split == 'train' else {'ID': f'{did}_{p}'})
                labels.append(row)
        (data / f'{split}.json').write_text(json.dumps(dialogs, ensure_ascii=False))
        with (data / f'y{split}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(labels[0]))
            w.writeheader(); w.writerows(labels)
    test = load_dataset(data, 'test')
    with (data / 'sample_submission.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['ID', 'is_bot'])
        w.writerows((r['ID'], 0.5) for r in reversed(test))
    train = load_dataset(data, 'train')
    membership = [dict(ID=r['ID'], dialog_id=r['dialog_id'], group=r['group'], fold=(i // 4 - 1))
                  for i, r in enumerate(train)]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    return data, train, test, membership, hashes
