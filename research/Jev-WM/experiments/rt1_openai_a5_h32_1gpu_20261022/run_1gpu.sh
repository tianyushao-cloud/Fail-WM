#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_openai_a5_h32_1gpu_20261022
if [[ -f "$TASK_ROOT/PAUSED_BY_USER" ]]; then
  echo "PAUSED_BY_USER: launch skipped; training remains paused"
  exit 0
fi
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$TASK_ROOT"
exec "$TASK_PYTHON" -u run.py "${1:-all}" --gpus 1
