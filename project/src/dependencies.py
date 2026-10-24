"""Installed-public-package closure; no pip freeze, installs or network access."""
from __future__ import annotations

import importlib.metadata as metadata
import os
import platform

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def read_pins(path):
    pins = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        req = Requirement(line)
        specs = list(req.specifier)
        if req.url or req.marker or req.extras or len(specs) != 1 or specs[0].operator != '==' or '*' in specs[0].version:
            raise ValueError('dependency files must contain only exact pins')
        name = canonicalize_name(req.name)
        if name in pins:
            raise ValueError('duplicate dependency pin')
        pins[name] = specs[0].version
    return pins


def installed_closure(roots):
    pending, packages, edges = list(roots), {}, {}
    environment: dict[str, str] = {str(k): str(v) for k, v in default_environment().items()}
    environment['extra'] = ''
    while pending:
        name = canonicalize_name(pending.pop())
        if name in packages:
            continue
        dist = metadata.distribution(name)
        packages[name] = dist.version
        edges[name] = []
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate(environment):
                continue
            if req.url or req.extras:
                raise ValueError('unexpected URL/extras dependency outside approved closure')
            dep = canonicalize_name(req.name)
            version = metadata.version(dep)
            if version not in req.specifier:
                raise ValueError(f'installed dependency constraint mismatch: {name} -> {dep}')
            edges[name].append({'name': dep, 'requirement': str(req)})
            pending.append(dep)
    for name, version in roots.items():
        if packages.get(name) != version:
            raise ValueError(f'root dependency version mismatch: {name}')
    return {'packages': dict(sorted(packages.items())), 'dependency_edges': dict(sorted(edges.items()))}


def check_environment(root):
    roots = read_pins(root / 'requirements.in')
    lock = read_pins(root / 'requirements.lock')
    closure = installed_closure(roots)
    if closure['packages'] != lock:
        raise ValueError('installed recursive dependency closure differs from exact lock')
    if platform.python_version() != '3.13.2':
        raise ValueError('frozen reproduction requires Python 3.13.2')
    threads = {key: os.environ.get(key) for key in THREAD_VARIABLES}
    if any(value != '4' for value in threads.values()):
        raise ValueError('all five BLAS/OpenMP thread variables must be 4; use run.sh')
    if not os.environ.get('TMPDIR'):
        raise ValueError('TMPDIR must be set by run.sh')
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                system=platform.system(), release=platform.release(), machine=platform.machine(),
                lock_scope='Resolved against installed metadata for CPython 3.13.2 / macOS arm64; not a universal wheel or cross-platform lock.',
                threads=threads, tmpdir_policy='Explicit writable scratch; absolute location intentionally not exported',
                sqlite_version=__import__('sqlite3').sqlite_version, **closure)
