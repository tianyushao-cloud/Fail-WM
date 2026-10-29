#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_openai_a5_h32_20261021
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
TASK_STAGE="${1:-all}"
case "$TASK_STAGE" in
  check|pilot|all) ;;
  *) echo "Usage: bash run_4gpu.sh [check|pilot|all]" >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$TASK_ROOT"
if [[ "$TASK_STAGE" != check ]]; then
  "$TASK_PYTHON" -u prepare_4gpu.py
fi
exec "$TASK_PYTHON" -u run.py "$TASK_STAGE" --gpus 4
