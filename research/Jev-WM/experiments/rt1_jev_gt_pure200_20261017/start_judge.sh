#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/experiments/rt1_jev_gt_pure200_20261017
TASK_PYTHON=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export JEV_HTTPS_PROXY="${JEV_HTTPS_PROXY:-http://127.0.0.1:17897}"
cd "$TASK_ROOT"
exec "$TASK_PYTHON" -u judge_service.py
