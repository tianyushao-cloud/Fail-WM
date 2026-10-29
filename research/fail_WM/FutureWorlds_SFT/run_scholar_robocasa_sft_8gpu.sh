#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
BUNDLE="$ROOT/assets/futureworlds_scholar_robocasa"
TEXT_MODEL="/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
GPU_COUNT="$($PYTHON -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[[ "$GPU_COUNT" -ge 8 ]] || { echo "Expected at least 8 CUDA devices; found $GPU_COUNT." >&2; exit 1; }

export PYTHONPATH="$FW_CODE/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

"$PYTHON" "$ROOT/prepare_scholar_robocasa_sft.py" --root "$ROOT"

for variant in success natural; do
  NAME="${variant}_scholar_sft_10k"
  OUTPUT="$ROOT/runs/$NAME"
  if [[ -f "$OUTPUT/complete.json" ]]; then
    echo "10k SFT already complete: $OUTPUT"
    continue
  fi
  case "$variant" in
    success) MANIFEST="$ROOT/robocasa_data/train_success.json" ;;
    natural) MANIFEST="$ROOT/robocasa_data/train_natural.json" ;;
  esac
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m futureworlds cache \
    --bundle "$BUNDLE" \
    --dataset robocasa \
    --manifest "$MANIFEST" \
    --output "$ROOT/cache/token_cache/$NAME" \
    --device cuda \
    --text-model "$TEXT_MODEL"
  cd "$FW_CODE"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m futureworlds.batched_sft --config "$ROOT/runs/$NAME.json"
done
