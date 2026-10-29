#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="${PYTHON:-/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python}"
MODEL="${MODEL:-/media/tyshao/fail_WM/FutureWorlds_workspace/assets/qwen2.5-vl-3b-instruct}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/evals_full}"
NAME="${1:?usage: $0 RUN_NAME}"
MAX_CASES="${MAX_CASES:-20}"

RUN_ROOT="$EVAL_ROOT/$NAME"
[[ -d "$RUN_ROOT/test" ]] || { echo "Missing full test rollout: $RUN_ROOT/test" >&2; exit 1; }
[[ -f "$RUN_ROOT/video_judge.json" ]] || { echo "Missing video judge report: $RUN_ROOT/video_judge.json" >&2; exit 1; }

exec "$PYTHON" "$ROOT/vlm_review_robocasa.py" \
  --model "$MODEL" \
  --prediction-root "$RUN_ROOT/test" \
  --manifest "$ROOT/robocasa_data/test.json" \
  --judge-json "$RUN_ROOT/video_judge.json" \
  --judge-section prediction_test \
  --max-cases "$MAX_CASES" \
  --frames 8 \
  --device cuda \
  --output "$RUN_ROOT/vlm_review.json"
