#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_memspo200_upstream_20261017
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$TASK_ROOT"
"$TASK_PYTHON" -u run.py "${1:-all}" --gpus 8 2>&1 | tee -a logs/all.log
