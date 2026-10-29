#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPUS=4
export OUTPUT="${OUTPUT:-$TASK_ROOT/runs/success_memspo_4gpu}"
exec bash "$TASK_ROOT/scripts/run_success_memspo.sh" "$@"
