#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="${PYTHON:-/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/robocasa_data}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/evals}"
OUTPUT="${OUTPUT:-$EVAL_ROOT/comparison/outcome_report}"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
[[ "$#" -gt 0 ]] || { echo "usage: $0 RUN_NAME [RUN_NAME ...]" >&2; exit 2; }

RUN_ARGS=()
for NAME in "$@"; do
  VAL="$EVAL_ROOT/$NAME/val/summary.json"
  TEST="$EVAL_ROOT/$NAME/test/summary.json"
  [[ -f "$VAL" ]] || { echo "Missing validation summary: $VAL" >&2; exit 1; }
  [[ -f "$TEST" ]] || { echo "Missing test summary: $TEST" >&2; exit 1; }
  RUN_ARGS+=(--run "$NAME:$VAL:$TEST")
done

exec "$PYTHON" "$ROOT/report_robocasa_outcomes.py" \
  --val-manifest "$DATA_ROOT/val.json" \
  --test-manifest "$DATA_ROOT/test.json" \
  "${RUN_ARGS[@]}" \
  --horizon "${HORIZON:-20}" \
  --metric "${METRIC:-lpips}" \
  --output "$OUTPUT"
