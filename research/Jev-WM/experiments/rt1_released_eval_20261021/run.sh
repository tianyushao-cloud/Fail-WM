#!/usr/bin/env bash
# Evaluate released FutureWorlds RT-1 variants with the identical protocol used for SFT/R0/Qwen arms.
set -euo pipefail
E=/media/tyshao/Jev-WM/experiments
B=$E/rt1_memspo200_upstream_20261017
O=$E/rt1_released_eval_20261021
P=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
export PYTHONPATH=$E/rt1_qwen_a5_h32_20261020/source/src USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
cd "$E/rt1_qwen_a5_h32_20261020/source"
for v in main; do
  for s in test validation; do
    "$P" -u -m futureworlds evaluate --bundle /media/tyshao/Jev-WM/paper_assets/futureworlds_rt1 --dataset rt1 \
      --variant "$v" --manifest "$B/data/$s/manifest.json" --output "$O/$v/$s" --device cuda \
      --text-model /media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base --horizons 10 20 32 --beam-width 4 \
      > "$O/logs/${v}_$s.log" 2>&1
  done
done
echo DONE > "$O/logs/DONE"
