#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/robocasa_data}"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-$ROOT/assets/futureworlds_bundle_rt1_init}"
TEXT_MODEL="${TEXT_MODEL:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/evals}"
GPUS="${GPUS:-8}"
HORIZON="${HORIZON:-20}"
NAME="${1:?usage: $0 CHECKPOINT_DIR RUN_NAME}"
CHECKPOINT="${2:?usage: $0 CHECKPOINT_DIR RUN_NAME}"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
[[ -d "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be a positive integer" >&2; exit 1; }

export PYTHONPATH="$FW_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

RUN_ROOT="$EVAL_ROOT/$NAME"
MODEL_BUNDLE="$RUN_ROOT/bundle"
mkdir -p "$RUN_ROOT"

"$PYTHON" -m futureworlds export-model \
  --bundle "$SOURCE_BUNDLE" \
  --dataset robocasa \
  --checkpoint "$CHECKPOINT" \
  --output "$MODEL_BUNDLE"

for SPLIT in val test; do
  MANIFEST="$DATA_ROOT/$SPLIT.json"
  OUT="$RUN_ROOT/$SPLIT"
  mkdir -p "$OUT"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    -m futureworlds evaluate \
    --bundle "$MODEL_BUNDLE" \
    --dataset robocasa \
    --variant sft \
    --manifest "$MANIFEST" \
    --output "$OUT" \
    --device cuda \
    --metric-device cuda \
    --text-model "$TEXT_MODEL" \
    --horizons 10 20 32 \
    --beam-width 4 \
    --memory full
done

"$PYTHON" "$ROOT/report_robocasa_outcomes.py" \
  --val-manifest "$DATA_ROOT/val.json" \
  --test-manifest "$DATA_ROOT/test.json" \
  --run "$NAME:$RUN_ROOT/val/summary.json:$RUN_ROOT/test/summary.json" \
  --horizon "$HORIZON" \
  --metric lpips \
  --output "$RUN_ROOT/outcome_report"

echo "Evaluation complete: $RUN_ROOT"
