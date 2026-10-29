#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/robocasa_data}"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-$ROOT/assets/futureworlds_bundle_rt1_init}"
TEXT_MODEL="${TEXT_MODEL:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/evals_full}"
GPUS="${GPUS:-8}"
RUN_JUDGE="${RUN_JUDGE:-0}"
CHECKPOINT="${1:?usage: $0 CHECKPOINT_DIR RUN_NAME}"
NAME="${2:?usage: $0 CHECKPOINT_DIR RUN_NAME}"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
[[ -d "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be a positive integer" >&2; exit 1; }
[[ "$RUN_JUDGE" == 0 || "$RUN_JUDGE" == 1 ]] || { echo "RUN_JUDGE must be 0 or 1" >&2; exit 1; }

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
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    -m futureworlds.evaluate_robocasa_full \
    --bundle "$MODEL_BUNDLE" \
    --manifest "$DATA_ROOT/$SPLIT.json" \
    --output "$RUN_ROOT/$SPLIT" \
    --device cuda \
    --metric-device cuda \
    --text-model "$TEXT_MODEL" \
    --beam-width 4 \
    --memory full
done

if [[ "$RUN_JUDGE" == 1 ]]; then
  "$PYTHON" "$ROOT/robocasa_video_judge.py" \
    --val-manifest "$DATA_ROOT/val.json" \
    --test-manifest "$DATA_ROOT/test.json" \
    --prediction-val "$RUN_ROOT/val" \
    --prediction-test "$RUN_ROOT/test" \
    --horizon 0 \
    --tail 5 \
    --output "$RUN_ROOT/video_judge.json"
fi

echo "Full rollout inference complete: $RUN_ROOT"
