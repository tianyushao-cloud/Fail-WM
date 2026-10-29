#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="/media/tyshao/fail_WM/FutureWorlds_workspace/code"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
export PYTHONPATH="$FW_CODE/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
"$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 8, "Expected 8 GPUs"'
"$PYTHON" prepare_followup_experiments.py codec
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  train_robocasa_codec_mixed.py --config "$ROOT/experiments_20261009/codec_mixed.json"
"$PYTHON" prepare_followup_experiments.py select_codec
