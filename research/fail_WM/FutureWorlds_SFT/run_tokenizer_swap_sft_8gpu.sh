#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
VARIANT="${1:-success}"
case "$VARIANT" in
  success)
    NAME="success_only_sft_new_codec_10k"
    CACHE_NAME="success_new_codec_10k"
    MANIFEST="train_success.json"
    ;;
  natural)
    NAME="natural_sft_new_codec_10k"
    CACHE_NAME="natural_new_codec_10k"
    MANIFEST="train_natural.json"
    ;;
  *) echo "usage: $0 [success|natural]" >&2; exit 2 ;;
esac
CONFIG="$ROOT/runs/$NAME.json"
CACHE="$ROOT/cache/token_cache/$CACHE_NAME"
OUTPUT="$ROOT/runs/$NAME"
TEXT_MODEL="/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
GPU_COUNT="$($PYTHON -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[[ "$GPU_COUNT" -ge 8 ]] || { echo "Expected at least 8 CUDA devices; found $GPU_COUNT." >&2; exit 1; }
if [[ -f "$OUTPUT/complete.json" ]]; then
  echo "10k SFT already complete: $OUTPUT"
  exit 0
fi

export PYTHONPATH="$FW_CODE/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

"$PYTHON" "$ROOT/prepare_tokenizer_swap_sft.py" --root "$ROOT" --variant "$VARIANT"

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m futureworlds cache \
  --bundle "$ROOT/assets/futureworlds_bundle_robocasa_codec_10k" \
  --dataset robocasa \
  --manifest "$ROOT/robocasa_data/$MANIFEST" \
  --output "$CACHE" \
  --device cuda \
  --text-model "$TEXT_MODEL"

cd "$FW_CODE"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m futureworlds.batched_sft --config "$CONFIG"
