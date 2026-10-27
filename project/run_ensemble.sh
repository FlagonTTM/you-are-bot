#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
# Same explicit scratch policy and interpreter isolation as the native wrapper.
export TMPDIR="${COURSEWORK_TMPDIR:-${TMPDIR:-$ROOT/outputs/.scratch}}"
mkdir -p -- "$TMPDIR"
exec "${PYTHON:-python3}" -E -P -B "$ROOT/run_ensemble.py" "$@"
