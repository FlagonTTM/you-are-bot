#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
# Honor an explicitly approved scratch directory. Outside Hermes, the fallback
# is project-owned outputs/.scratch, never a hardcoded machine-specific path.
export TMPDIR="${COURSEWORK_TMPDIR:-${TMPDIR:-$ROOT/outputs/.scratch}}"
mkdir -p -- "$TMPDIR"
# -E -P ignores ambient PYTHONPATH/current directory while retaining installed
# user-site packages (this offline environment installs public dependencies there).
# run.py adds only this project's src for application-code imports.
exec "${PYTHON:-python3}" -E -P -B "$ROOT/run.py" "$@"
