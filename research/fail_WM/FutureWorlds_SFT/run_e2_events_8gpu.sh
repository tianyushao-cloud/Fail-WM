#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FW_CODE="/media/tyshao/fail_WM/FutureWorlds_workspace/code"
PYTHON="${PYTHON:-$FW_CODE/.venv/bin/python}"
export PYTHONPATH="$FW_CODE/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
[[ -f "$ROOT/evals_e1_expanded_20261009/complete_success_natural.json" ]] || { echo 'Expanded E1 baseline must finish before E2.' >&2; exit 1; }
"$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 8, "Expected 8 GPUs"'
for variant in success natural; do
  config="$ROOT/experiments_20261009/e2_${variant}.json"
  [[ -f "$config" ]] || { echo "Missing prepared config: $config" >&2; exit 1; }
  # Use a separate runtime config for resume; keep the reviewed protocol immutable.
  runtime="$ROOT/experiments_20261009/e2_${variant}_runtime.json"
  status="$($PYTHON - "$config" "$runtime" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());out=Path(c['output'])
if (out/'complete.json').exists():
    print('complete')
else:
    checkpoints=sorted((out/'checkpoints').glob('step_*/experiment.json'))
    if checkpoints:c['resume']=str(checkpoints[-1].parent)
    Path(sys.argv[2]).write_text(json.dumps(c,indent=2)+'\n')
    print('run')
PY
)"
  [[ "$status" != complete ]] || continue
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m futureworlds.batched_sft --config "$runtime"
done
