#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
BUNDLE="${BUNDLE:-$TASK_ROOT/assets/local_success_sft}"
DATA_ROOT="${DATA_ROOT:-/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data}"
TEXT_MODEL="${TEXT_MODEL:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base}"
TASK_DEFAULT_PYTHON="/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python"
if [[ -x "$TASK_ROOT/.venv-cu124/bin/python" ]]; then TASK_DEFAULT_PYTHON="$TASK_ROOT/.venv-cu124/bin/python"; fi
TASK_PYTHON="${PYTHON:-$TASK_DEFAULT_PYTHON}"
TASK_OUTPUT="${OUTPUT:-$TASK_ROOT/runs/local_success_memspo_h32}"
TASK_STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONPATH="$TASK_ROOT/vendor/scholar_futureworlds/src"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$TASK_OUTPUT/logs"
echo "Stage=$TASK_STAGE; original episode first two frames; local success-only SFT+codec; released MemSPO/R0; 4 GPUs; horizon32; 200 updates; validation checkpoint selection; output=$TASK_OUTPUT"
"$TASK_PYTHON" -u scripts/local_success_memspo.py "$TASK_STAGE" \
 --bundle "$BUNDLE" --data-root "$DATA_ROOT" --text-model "$TEXT_MODEL" \
 --output "$TASK_OUTPUT" --gpus 4 --horizon 32 "$@" 2>&1 | tee -a "$TASK_OUTPUT/logs/$TASK_STAGE.log"

