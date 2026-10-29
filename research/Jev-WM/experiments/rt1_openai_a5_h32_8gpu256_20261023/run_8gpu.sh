#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_openai_a5_h32_8gpu256_20261023
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
TASK_STAGE="${1:-all}"
case "$TASK_STAGE" in
  check|prepare|train|all|finish) ;;
  *) echo "Usage: bash run_8gpu.sh [check|prepare|train|all|finish]" >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "$TASK_ROOT"
exec "$TASK_PYTHON" -u run.py "$TASK_STAGE"
