# RoboCasa SFT preparation

`prepare_robocasa_sft.py` converts the audited RoboCasa HDF5 episodes into the
native FutureWorlds format. It keeps every fourth control step, packs four
12-D raw controls plus a zero padding value into one 52-D action row, resizes
RGB with Pillow BICUBIC from `480x640` to `256x320`, and writes a zero final
action row. The instruction is constant and contains no outcome label.

The generated root contains:

```text
robocasa_data/
  cases/*.npz
  train_natural.json       # 443 success + 357 failure
  train_success.json       # success-only control
  train_balanced.json      # 357 success + 357 failure
  val.json                 # held-out initial-state groups
  test.json                # held-out initial-state groups
```

Outcome and failure type are manifest metadata only. The model reads only RGB,
packed actions, and the constant instruction.

## Required assets before launch

The source repository does not contain a complete RoboCasa FutureWorlds bundle.
Place the bundle at `assets/futureworlds_bundle` or set `FW_BUNDLE`. It must
include a verified `manifest.json`, `robocasa/codec/`,
`robocasa/preprocessing/action_ranges.pt`, and
`robocasa/models/sft/` with the predictor config, weights, and text adapters.
The RT-1 base checkpoint can be supplied through `INITIAL_CHECKPOINT`; the
launcher verifies its model hash against the RT-1 upstream SHA recorded in the
template and initializes missing text adapters for this SFT run.

On the 8-card host, install the FutureWorlds environment in the checked-out
code directory, using the host's CUDA-compatible PyTorch build:

```bash
cd /media/tyshao/fail_WM/FutureWorlds_workspace/code
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[full]'
```

## 8-card launch

After the bundle is present, the default natural-mix run is:

```bash
cd /media/tyshao/fail_WM/FutureWorlds_SFT
./run_sft_8gpu.sh natural
```

The other controlled runs use the same command with `success` or `balanced`.
The pilot is 10,000 optimizer steps, global batch 64, base LR `1e-5`, text
adapter LR `1e-4`, BF16 autocast, and checkpoints every 500 steps. All three
runs use the same validation manifest and seed; only the training manifest and
output directory differ. The test manifest is reserved for the final outcome
probe and must not be used for checkpoint selection.

To point at a different machine layout, set `FW_CODE`, `DATA_ROOT`,
`FW_BUNDLE`, `TEXT_MODEL`, `INITIAL_CHECKPOINT`, `OUTPUT_ROOT`, and `PYTHON`
before invoking the launcher.
