#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/media/tyshao/fail_WM/FutureWorlds_SFT
FW_CODE=/media/tyshao/fail_WM/FutureWorlds_workspace/code
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
EXP="$ROOT/experiments_e2v2_adrop_20261012"
OUT="$ROOT/evals_2x2_20261012"
export PYTHONPATH="$FW_CODE/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/media/tyshao/cache/torch
cd "$ROOT"
mkdir -p "$EXP/logs"
exec 9>"$EXP/submit.lock"
flock -n 9 || { echo 'This 2x2 job is already running.' >&2; exit 1; }
LOG="$EXP/logs/job_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG") 2>&1
trap 'status=$?; echo "FAILED: line $LINENO, exit $status; log: $LOG"; exit "$status"' ERR
"$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 8, "Requires 8 visible GPUs"'
"$PYTHON" prepare_2x2_20261012.py preflight
"$PYTHON" smoke_2x2_20261012.py --memory-gib 8
for label in B C D; do
  state="$("$PYTHON" prepare_2x2_20261012.py runtime --label "$label")"
  if [[ "$state" == complete ]]; then
    echo "$label already complete and verified"
    continue
  fi
  [[ "$state" == run ]]
  if [[ "$label" == B ]]; then
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      -m futureworlds.batched_sft --config "$EXP/${label}_runtime.json"
  else
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      "$ROOT/batched_sft_adrop_20261012.py" --config "$EXP/${label}_runtime.json"
  fi
done
"$PYTHON" prepare_2x2_20261012.py verify-training
"$PYTHON" prepare_2x2_20261012.py export
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/handoff_2x2_worker_20261012.py"
"$PYTHON" "$EXP/gt_handoff_frozen.py" report --out "$OUT"
# A event evaluation belongs to Opus. Only A's fixed-horizon LPIPS baseline is added here.
for label in A B C D; do
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m futureworlds evaluate --bundle "$OUT/bundles/$label" --dataset robocasa --variant sft \
    --manifest "$ROOT/robocasa_data/val.json" --output "$OUT/frame_metrics/$label" \
    --device cuda --metric-device cuda \
    --text-model /media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base \
    --horizons 10 20 --beam-width 4 --memory full
done
"$PYTHON" report_2x2_20261012.py
echo "COMPLETE: $OUT/COMPARISON_zh.md"
