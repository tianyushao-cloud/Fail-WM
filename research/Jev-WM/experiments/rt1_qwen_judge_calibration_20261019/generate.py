"""Stage 1 (WM only, no judge): candidates and controls for judge calibration.

Validation windows only; zero optimizer steps. Candidates use the same MemSPO
group beam (4 groups x 2 beams, horizon 10, diversity 0.1) as RL training.
LPIPS/MAE are computed here as offline diagnostics only, never as reward.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "rt1_jev_gt_pure200_20261017" / "source"
BASE = ROOT.parent / "rt1_memspo200_upstream_20261017"
sys.path.insert(0, str(SOURCE / "src"))
from futureworlds.batch import TensorBatch  # noqa: E402
from futureworlds.data import read_case, read_manifest  # noqa: E402
from futureworlds.pipeline import WorldPipeline  # noqa: E402
from futureworlds.rewards import CodecWorker  # noqa: E402
from futureworlds.search import rollout_group_beam  # noqa: E402



def rgb(x):
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite RGB")
    return x.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def repeats(clip):
    return int(sum(np.array_equal(a, b) for a, b in zip(clip[:-1], clip[1:])))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="sft", choices=["sft", "r0rl"])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--name", default=None, help="output folder under cases/ (default: variant)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    cfg = json.loads((BASE / "train.json").read_text())
    torch.manual_seed(cfg["seed"])
    bundle, variant = (cfg["bundle"], "sft") if args.variant == "sft" else (str(BASE / "rl_bundle"), "main")
    pipe = WorldPipeline(bundle, "rt1", variant, "cuda:0", cfg["text_model"])
    pipe.model.eval().requires_grad_(False)
    worker = CodecWorker(pipe.codec, None)
    import lpips

    perceptual = lpips.LPIPS(net="vgg").cuda().eval()
    H = args.horizon
    manifest = BASE / f"data/{args.split}/manifest.json"
    rows = read_manifest(manifest, "rt1")["cases"][args.offset : args.offset + args.limit]
    out = ROOT / "cases" / (args.name or args.variant)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, row in enumerate(rows):
        images, actions, text = read_case(manifest, row, pipe.action_dim, H)
        # encode_training_window also sets the task text on pipe.model.
        ctx, dynamics, acts, pixels, _ = pipe.encode_training_window(images, actions, text)
        frames, _ = rollout_group_beam(
            pipe.model, ctx.reshape(1, -1) + 4375, dynamics[1:2], acts, H, penalty=0.1, groups=4, beams=2
        )
        codes = torch.stack([f["response"] for f in frames], 1)
        decode = lambda c: worker.detokenize(  # noqa: E731
            TensorBatch(dict(tokens=c, ctx_tokens=ctx.expand(len(c), -1, -1))), None
        ).batch["pixels"][:, 1:]
        pred = decode(codes)
        gt_codec = decode(dynamics[2 : 2 + H][None])[0]
        raw = pixels[0, 2 : 2 + H]
        lp = [float(perceptual(p * 2 - 1, raw * 2 - 1).mean()) for p in pred]
        mae = [float((p - raw).abs().mean()) for p in pred]
        predictions = np.stack([rgb(p) for p in pred])
        record = dict(
            index=i,
            id=row["id"],
            task=text,
            variant=args.variant,
            lpips=lp,
            mae=mae,
            frame_repeats=[repeats(p) for p in predictions],
            token_repeats=[int(sum(torch.equal(a, b) for a, b in zip(c[:-1], c[1:]))) for c in codes],
            gt_codec_mae=float((gt_codec - raw).abs().mean()),
            gt_codec_lpips=float(perceptual(gt_codec * 2 - 1, raw * 2 - 1).mean()),
        )
        np.savez_compressed(
            out / f"{i:03d}.npz",
            initial=rgb(pixels[0, :2]),
            reference_raw=rgb(raw),
            reference_codec=rgb(gt_codec),
            predictions=predictions,
            tokens=codes.cpu().numpy().astype(np.int16),
            task=np.array(text),
        )
        index.append(record)
        print(json.dumps(record), flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
