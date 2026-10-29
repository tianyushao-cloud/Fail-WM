#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="${FW_CODE:-/media/tyshao/fail_WM/FutureWorlds_workspace/code}"
DATA_ROOT="${DATA_ROOT:-$ROOT/robocasa_data}"
FW_BUNDLE="${FW_BUNDLE:-$ROOT/assets/futureworlds_bundle_robocasa_codec}"
TEXT_MODEL="${TEXT_MODEL:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/rt1-upstream/model}"
OUTPUT="${OUTPUT:-$ROOT/runs/paper_success_sft_50k}"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
CONFIG="$ROOT/runs/paper_success_sft_50k.json"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
GPU_COUNT="$($PYTHON -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[[ "$GPU_COUNT" -ge 8 ]] || { echo "Expected at least 8 CUDA devices; found $GPU_COUNT." >&2; exit 1; }
[[ -f "$FW_BUNDLE/manifest.json" ]] || { echo "Missing adapted bundle: $FW_BUNDLE" >&2; exit 1; }
[[ -f "$DATA_ROOT/train_success.json" && -f "$DATA_ROOT/val.json" ]] || { echo "Missing RoboCasa manifests" >&2; exit 1; }

mkdir -p "$ROOT/runs"
cat > "$CONFIG" <<EOF
{
  "dataset": "robocasa",
  "method": "sft",
  "bundle": "$FW_BUNDLE",
  "text_model": "$TEXT_MODEL",
  "train_manifest": "$DATA_ROOT/train_success.json",
  "evaluation_manifest": "$DATA_ROOT/val.json",
  "output": "$OUTPUT",
  "initial_variant": "sft",
  "device": "cuda",
  "seed": 20261007,
  "steps": 50000,
  "global_batch": 64,
  "lr": 1e-5,
  "adapter_lr": 1e-4,
  "warmup_steps": 500,
  "horizon": 10,
  "sft_bf16": true,
  "checkpoint_every": 5000,
  "paper_locked": false,
  "initial_checkpoint": "$INITIAL_CHECKPOINT",
  "initial_sha256": "e5229f8a9de990d61cd900d05e6aaf5909a83e3634c5695d46ec8a46568dfe37",
  "initialize_text_adapters": true
}
EOF

export PYTHONPATH="$FW_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd "$FW_CODE"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m futureworlds train --config "$CONFIG"
