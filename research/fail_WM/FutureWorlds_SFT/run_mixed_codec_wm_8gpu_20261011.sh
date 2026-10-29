#!/usr/bin/env bash
# Fresh codec caches, then success-only and mixed WMs, using the frozen codec.
set -euo pipefail
ROOT="/media/tyshao/fail_WM/FutureWorlds_SFT"
FW_CODE="/media/tyshao/fail_WM/FutureWorlds_workspace/code"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
EXP="$ROOT/experiments_mixed_codec_wm_20261011"
export PYTHONPATH="$FW_CODE/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
exec 9>"$EXP/train.lock"
flock -n 9 || { echo 'This WM experiment is already running in another job.' >&2; exit 1; }
"$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 8, "This job requires 8 GPUs"'
"$PYTHON" prepare_mixed_codec_wm_20261011.py preflight
for variant in success natural; do
  status="$("$PYTHON" prepare_mixed_codec_wm_20261011.py runtime --variant "$variant")"
  [[ "$status" != complete ]] || continue
  config="$EXP/${variant}_runtime.json"
  mapfile -t fields < <("$PYTHON" - "$config" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
for k in ('bundle','train_manifest','token_cache','text_model'):
    print(c[k])
PY
)
  [[ "${#fields[@]}" -eq 4 ]] || { echo 'Invalid experiment configuration' >&2; exit 1; }
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m futureworlds cache --bundle "${fields[0]}" --dataset robocasa \
    --manifest "${fields[1]}" --output "${fields[2]}" --device cuda --text-model "${fields[3]}"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m futureworlds.batched_sft --config "$config"
done
echo 'Both 10000-step WM runs completed.'
