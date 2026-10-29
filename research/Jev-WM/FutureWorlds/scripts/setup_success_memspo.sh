#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_BASE_PYTHON="${BASE_PYTHON:-python3}"
"$TASK_BASE_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12 for this pinned environment"'
TASK_ENV="$TASK_ROOT/.venv-cu124"
if [[ ! -x "$TASK_ENV/bin/python" ]]; then
  "$TASK_BASE_PYTHON" -m venv "$TASK_ENV"
fi
TASK_PYTHON="$TASK_ENV/bin/python"
"$TASK_PYTHON" -m pip install --upgrade pip
"$TASK_PYTHON" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$TASK_PYTHON" -m pip install -r requirements-cu124.txt
"$TASK_PYTHON" -m pip install --no-deps -e .
"$TASK_PYTHON" -m pip check
USE_TF=0 USE_FLAX=0 "$TASK_PYTHON" - <<'PY'
import torch, torchvision, transformers, diffusers, lpips, piqa
assert torch.version.cuda == "12.4", f"Expected cu124, got {torch.__version__}"
assert torch.cuda.is_available(), "CUDA unavailable: check host driver and container GPU passthrough"
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        x = torch.ones((32, 32), device=f"cuda:{index}")
        assert (x @ x).sum().item() == 32768
        torch.cuda.synchronize()
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
print(f"torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA={torch.version.cuda}, cuDNN={torch.backends.cudnn.version()}")
PY
echo "Environment ready. Run: bash scripts/run_success_memspo.sh all"
