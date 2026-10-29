#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_memspo_20261017
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
export PYTHONPATH=/media/tyshao/Jev-WM/FutureWorlds/vendor/scholar_futureworlds/src
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd "$TASK_ROOT"
if [[ -f stopped_by_user.json ]]; then
  echo "Stopped by user: single-GPU / 32-frame protocol is invalid. Do not resume this experiment." >&2
  exit 2
fi
exec "$TASK_PYTHON" -u scripts/experiment.py "${1:-all}"
