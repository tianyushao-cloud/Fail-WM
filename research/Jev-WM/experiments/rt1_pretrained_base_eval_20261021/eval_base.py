"""Evaluate the pretrained-only RLVR-World RT-1 base (no FutureWorlds SFT, no RL) with the identical
FutureWorlds pipeline: same codec, action ranges, ReCAP memory (anchor1+recent6), per-frame beam4,
no GT refresh, same metrics.

The base has no text adapters. FutureWorlds adapters are zero-initialized residuals (out.weight = 0),
so loading the base core with freshly initialized adapters is exactly the base model with no text.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

import futureworlds.pipeline as pipeline_module
from futureworlds import evaluation
from futureworlds.checkpoints import load_world_model as _load

BASE = "/media/tyshao/Jev-WM/paper_assets/rlvr_world/rt1-world-model-multi-step-base"


def load_base(root, dataset, variant="sft", device="cpu", verify=True):
    model = _load(root, dataset, "sft", "cpu", verify)  # architecture + adapter modules only
    state = load_file(str(Path(BASE) / "model.safetensors"))
    model.core.load_state_dict(state, strict=True)
    for adapter in model.adapters.values():
        torch.nn.init.zeros_(adapter.out.weight)
    assert all(float(a.out.weight.abs().max()) == 0.0 for a in model.adapters.values())
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    pipeline_module.load_world_model = load_base
    kwargs = dict(bundle="/media/tyshao/Jev-WM/paper_assets/futureworlds_rt1", dataset="rt1", variant="sft",
                  manifest=args.manifest, output=args.output, device="cuda",
                  text_model="/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base",
                  horizons=[10, 20, 32], beam_width=4)
    if args.limit:
        kwargs["limit"] = args.limit
    evaluation.evaluate(**kwargs)
    marker = Path(args.output) / "WEIGHTS.txt"
    marker.write_text(f"core weights: {BASE} (thuml/rt1-world-model-multi-step-base @ 242d6b7)\n"
                      "text adapters: zero (no text conditioning); protocol 'variant' field says sft only because "
                      "the bundle config/codec/action ranges are reused.\n")


if __name__ == "__main__":
    main()
