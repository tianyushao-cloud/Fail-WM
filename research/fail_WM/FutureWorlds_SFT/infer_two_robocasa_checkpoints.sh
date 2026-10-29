#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
SUCCESS="$ROOT/runs/success_only_sft_cached_batched/checkpoints/step_010000"
NATURAL="$ROOT/runs/natural_sft_cached_batched/checkpoints/step_010000"

for checkpoint in "$SUCCESS" "$NATURAL"; do
  [[ -f "$checkpoint/experiment.json" && -f "$checkpoint/model.safetensors" && -f "$checkpoint/text_adapters.safetensors" ]] || {
    echo "Incomplete checkpoint: $checkpoint" >&2
    exit 1
  }
done

"$ROOT/evaluate_robocasa_full_checkpoint.sh" "$SUCCESS" success_step10000_full
"$ROOT/evaluate_robocasa_full_checkpoint.sh" "$NATURAL" natural_step10000_full

echo "Both checkpoints inferred under $ROOT/evals_full"
