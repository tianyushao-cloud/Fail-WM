#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
# Included source/assets are the exact runtime snapshot used for these data.
uv pip install --python .venv/bin/python --no-deps -e vendor/robosuite -e vendor/robocasa
