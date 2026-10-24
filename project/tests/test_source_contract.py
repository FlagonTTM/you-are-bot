"""Unit fixtures only; these tests are not official-data model evidence."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def test_frozen_source_copies_and_recipe():
    assert (ROOT / 'src/origins.json').is_file(), 'missing frozen-source manifest'
    origins = json.loads((ROOT / 'src/origins.json').read_text())
    for name, source in origins['files'].items():
        assert hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest() == source['sha256']
        assert not Path(source['origin']).is_absolute()
    experiment = importlib.import_module('experiment')
    expected = dict(iterations=1000, depth=5, learning_rate=0.04, l2_leaf_reg=5,
                    loss_function='Logloss', random_seed=1729, thread_count=4,
                    verbose=False, allow_writing_files=False)
    assert experiment.BEHAVIOR_CONFIG == expected
    config = json.loads((ROOT / 'configs/template.json').read_text())
    assert config['model_params'] == expected
    assert config['feature_counts'] == {'numeric': 162, 'categorical': 14}
    assert config['historical_oof_logloss'] == 0.16919850572003053
