#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
CODEC_RUN="$ROOT/runs/paper_adapt_success_codec"
if [[ ! -f "$CODEC_RUN/complete.json" ]]; then
  "$ROOT/run_paper_robocasa_codec_8gpu.sh"
fi
"$ROOT/run_tokenizer_swap_sft_8gpu.sh" success
"$ROOT/run_tokenizer_swap_sft_8gpu.sh" natural
