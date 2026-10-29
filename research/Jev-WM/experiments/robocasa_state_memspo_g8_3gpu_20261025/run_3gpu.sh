#!/usr/bin/env bash
# Usage: bash run_3gpu.sh check | bash run_3gpu.sh train
# Re-run train to resume the last complete 3-GPU checkpoint.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${TASK_PYTHON:-/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python}"
cd "$TASK_ROOT"
mkdir -p logs
"$TASK_PYTHON" -u run.py "${1:-train}" 2>&1 | tee -a logs/train.log
