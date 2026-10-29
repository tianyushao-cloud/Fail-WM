#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_openai_a5_h32_20261021
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export JEV_HTTPS_PROXY="${JEV_HTTPS_PROXY:-http://127.0.0.1:17897}"
cd "$TASK_ROOT"
exec "$TASK_PYTHON" -u run.py "${1:-all}" --gpus "${GPUS:-1}"
