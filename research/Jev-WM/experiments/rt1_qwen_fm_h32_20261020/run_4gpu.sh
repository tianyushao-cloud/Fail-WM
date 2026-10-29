#!/usr/bin/env bash
# Usage: bash run_4gpu.sh [check|all]   (set GPUS=8 for 8 cards; global batch stays 16)
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_qwen_fm_h32_20261020
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$TASK_ROOT"
"$TASK_PYTHON" -u run.py "${1:-all}" --gpus "${GPUS:-4}" 2>&1 | tee -a logs/all.log
