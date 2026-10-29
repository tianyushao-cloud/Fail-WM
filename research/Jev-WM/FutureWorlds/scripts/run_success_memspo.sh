#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
BUNDLE="${BUNDLE:-/media/tyshao/fail_WM/FutureWorlds_SFT/assets/futureworlds_scholar_robocasa}"
DATA_ROOT="${DATA_ROOT:-/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data}"
TEXT_MODEL="${TEXT_MODEL:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base}"
TASK_DEFAULT_PYTHON="/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python"
if [[ -x "$TASK_ROOT/.venv-cu124/bin/python" ]]; then
  TASK_DEFAULT_PYTHON="$TASK_ROOT/.venv-cu124/bin/python"
fi
TASK_PYTHON="${PYTHON:-$TASK_DEFAULT_PYTHON}"
if [[ ! -x "$TASK_PYTHON" ]] && ! command -v "$TASK_PYTHON" >/dev/null 2>&1; then
  echo "GPU Python not found. Run: bash scripts/setup_success_memspo.sh" >&2
  exit 1
fi
TASK_STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONPATH="$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
TASK_OUTPUT="${OUTPUT:-$TASK_ROOT/runs/success_memspo}"
mkdir -p "$TASK_OUTPUT/logs"
"$TASK_PYTHON" -u scripts/success_memspo.py "$TASK_STAGE" \
  --bundle "$BUNDLE" --data-root "$DATA_ROOT" --text-model "$TEXT_MODEL" \
  --output "$TASK_OUTPUT" --gpus "${GPUS:-8}" --steps "${STEPS:-200}" "${@}" \
  2>&1 | tee -a "$TASK_OUTPUT/logs/$TASK_STAGE.log"
