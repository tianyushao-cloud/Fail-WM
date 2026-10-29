#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/media/tyshao/Jev-WM/experiments/rt1_qwen_a5_h32_20261020/source/src USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
cd /media/tyshao/Jev-WM/experiments/rt1_pretrained_base_eval_20261021
for s in test validation; do
  /media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python -u eval_base.py --manifest /media/tyshao/Jev-WM/experiments/rt1_memspo200_upstream_20261017/data/$s/manifest.json --output /media/tyshao/Jev-WM/experiments/rt1_pretrained_base_eval_20261021/base/$s > logs/base_$s.log 2>&1
done
echo DONE > logs/DONE
