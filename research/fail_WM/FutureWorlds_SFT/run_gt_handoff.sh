#!/usr/bin/env bash
# GT-history handoff experiment; resumable (finished model x case JSONs are skipped).
set -euo pipefail
cd /media/tyshao/fail_WM/FutureWorlds_SFT
export PYTHONPATH=/media/tyshao/fail_WM/FutureWorlds_workspace/code/src:$PWD USE_TF=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
PY=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
SHARDS=${SHARDS:-2}
[[ -f evals_gt_handoff_20261010/selection.json ]] || $PY gt_handoff.py select
for i in $(seq 0 $((SHARDS-1))); do
  $PY gt_handoff.py run --shard $i --shards $SHARDS "$@" > evals_gt_handoff_20261010/run_shard$i.log 2>&1 &
done
wait
$PY gt_handoff.py report
