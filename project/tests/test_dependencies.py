import importlib
from pathlib import Path


def test_lock_is_exact_installed_public_recursive_closure():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'src/dependencies.py').exists(), 'missing recursive dependency closure'
    deps = importlib.import_module('dependencies')
    lock = deps.read_pins(root / 'requirements.lock')
    roots = deps.read_pins(root / 'requirements.in')
    closure = deps.installed_closure(roots)
    assert closure['packages'] == lock
    assert roots.keys() <= lock.keys()
    assert 'pytest' in lock and 'packaging' in lock
    assert all(version and '*' not in version for version in lock.values())
    info = deps.check_environment(root)
    assert info['packages'] == lock
    assert info['python'] == '3.13.2'
    assert info['dependency_edges']
