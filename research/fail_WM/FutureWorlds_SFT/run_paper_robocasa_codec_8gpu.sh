#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/robocasa_codec_paper_adapt.json}"
[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
GPU_COUNT="$($PYTHON -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[[ "$GPU_COUNT" -ge 8 ]] || { echo "Expected at least 8 CUDA devices; found $GPU_COUNT." >&2; exit 1; }
export PYTHONPATH="$FW_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd "$FW_CODE"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/train_robocasa_codec_adapt.py" --config "$CONFIG"
