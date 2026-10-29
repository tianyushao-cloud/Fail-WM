#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
DATA_ROOT="${DATA_ROOT:-$ROOT/robocasa_data}"
FW_BUNDLE="${FW_BUNDLE:-$ROOT/assets/futureworlds_bundle_rt1_init}"
TEXT_MODEL="${TEXT_MODEL:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/rt1-upstream/model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
VARIANT="${1:-natural}"

case "$VARIANT" in
  natural) TRAIN_MANIFEST="$DATA_ROOT/train_natural.json"; NAME="natural_sft" ;;
  success) TRAIN_MANIFEST="$DATA_ROOT/train_success.json"; NAME="success_only_sft" ;;
  balanced) TRAIN_MANIFEST="$DATA_ROOT/train_balanced.json"; NAME="balanced_sft" ;;
  *) echo "usage: $0 [natural|success|balanced]" >&2; exit 2 ;;
esac

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
GPU_COUNT="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[[ "$GPU_COUNT" -ge 8 ]] || { echo "Expected at least 8 CUDA devices; found $GPU_COUNT. Set PYTHON to the 8-card host environment." >&2; exit 1; }
[[ -f "$FW_BUNDLE/manifest.json" ]] || { echo "Missing FutureWorlds bundle: $FW_BUNDLE" >&2; exit 1; }
[[ -f "$TRAIN_MANIFEST" ]] || { echo "Missing data manifest: $TRAIN_MANIFEST" >&2; exit 1; }
[[ -f "$DATA_ROOT/val.json" ]] || { echo "Missing validation manifest: $DATA_ROOT/val.json" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
CONFIG="$OUTPUT_ROOT/${NAME}.json"
sed \
  -e "s|__FW_BUNDLE__|$FW_BUNDLE|g" \
  -e "s|__DATA_ROOT__/train_natural.json|$TRAIN_MANIFEST|g" \
  -e "s|__DATA_ROOT__/val.json|$DATA_ROOT/val.json|g" \
  -e "s|__OUTPUT_ROOT__/natural_sft|$OUTPUT_ROOT/$NAME|g" \
  -e "s|__INITIAL_CHECKPOINT__|$INITIAL_CHECKPOINT|g" \
  -e "s|__TEXT_MODEL__|$TEXT_MODEL|g" \
  "$ROOT/robocasa_sft_8gpu.template.json" > "$CONFIG"

cd "$FW_CODE"
export PYTHONPATH="$FW_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m futureworlds train --config "$CONFIG"
