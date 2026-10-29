#!/usr/bin/env bash
# codec34k WMs: open-loop (start) + late handoff (+2) only; resumable
set -euo pipefail
cd /media/tyshao/fail_WM/FutureWorlds_SFT
export PYTHONPATH=/media/tyshao/fail_WM/FutureWorlds_workspace/code/src:$PWD USE_TF=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
PY=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
B=$PWD/previews_wm_codec34k_20261012
for i in 0 1; do
  $PY gt_handoff.py run --out $PWD/evals_gt_handoff_codec34k_20261012 --labels start +2 --shard $i --shards 2 --model c34_success=$B/success_bundle --model c34_natural=$B/natural_bundle > evals_gt_handoff_codec34k_20261012/run_shard$i.log 2>&1 &
done
wait
$PY gt_handoff.py report --out $PWD/evals_gt_handoff_codec34k_20261012
