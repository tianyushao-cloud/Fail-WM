#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/media/tyshao/Jev-WM/FutureWorlds
TASK_PYTHON="${PYTHON:-/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python}"
if [[ -z "${PYTHON:-}" && -x "$TASK_ROOT/.venv-cu124/bin/python" ]]; then
 TASK_PYTHON="$TASK_ROOT/.venv-cu124/bin/python"
fi
export PYTHONPATH="$TASK_ROOT/vendor/scholar_futureworlds/src"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
cd "$TASK_ROOT"
mkdir -p runs/short_success_memspo/logs
TASK_STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
"$TASK_PYTHON" -u scripts/short_success_memspo.py "$TASK_STAGE" --gpus 8 "$@" 2>&1 | tee -a "runs/short_success_memspo/logs/$TASK_STAGE.log"

