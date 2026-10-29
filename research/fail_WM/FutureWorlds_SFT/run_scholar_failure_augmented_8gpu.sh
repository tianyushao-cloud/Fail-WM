#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
NAME="failure_augmented70_scholar_sft_10k"
BUNDLE="$ROOT/assets/futureworlds_scholar_robocasa"
TEXT_MODEL="/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
GPU_COUNT="$($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
[[ "$GPU_COUNT" -ge 8 ]] || { echo "Expected 8 CUDA devices; found $GPU_COUNT" >&2; exit 1; }

export PYTHONPATH="$FW_CODE/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

if [[ ! -f "$ROOT/runs/$NAME.json" ]]; then
  "$PYTHON" "$ROOT/prepare_scholar_failure_augmented.py" --root "$ROOT"
fi
if [[ -f "$ROOT/runs/$NAME/complete.json" ]]; then
  echo "SFT already complete: $ROOT/runs/$NAME"
  exit 0
fi
if [[ ! -f "$ROOT/cache/token_cache/$NAME/complete.json" ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m futureworlds cache \
    --bundle "$BUNDLE" \
    --dataset robocasa \
    --manifest "$ROOT/robocasa_data/train_scholar_failure_augmented.json" \
    --output "$ROOT/cache/token_cache/$NAME" \
    --device cuda \
    --text-model "$TEXT_MODEL"
fi
cd "$FW_CODE"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m futureworlds.batched_sft --config "$ROOT/runs/$NAME.json"
