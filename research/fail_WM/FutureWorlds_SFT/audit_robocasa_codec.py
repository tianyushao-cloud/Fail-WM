#!/usr/bin/env python3
"""Measure the frozen RoboCasa codec's reconstruction on the local task."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from futureworlds.codec import CompressiveVQModelFSQ


def score(real, reconstructed):
    a = real.astype(np.float32) / 255
    b = reconstructed.astype(np.float32)
    return {
        "mae_255": float(np.abs(a - b).mean() * 255),
        "psnr": float(-10 * np.log10(max(np.square(a - b).mean(), 1e-12))),
        "horizontal_gradient_ratio": float(
            np.abs(np.diff(b, axis=1)).mean()
            / max(np.abs(np.diff(a, axis=1)).mean(), 1e-12)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest["cases"][: args.limit]
    codec = CompressiveVQModelFSQ.from_pretrained(args.codec, local_files_only=True).to("cuda").eval()
    results = []
    image = None
    with torch.inference_mode():
        for row in rows:
            with np.load(Path(args.manifest).parent / row["file"], allow_pickle=False) as data:
                frames = data["images"].copy()
            for position in (1, len(frames) - 1):
                pair = frames[[0, position]]
                x = torch.from_numpy(pair).permute(0, 3, 1, 2)[None].float().cuda() / 255
                context, dynamics = codec.tokenize(x.contiguous())
                reconstructed = codec.detokenize(context, dynamics)[0, 1]
                reconstructed = reconstructed.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
                metrics = score(pair[1], reconstructed)
                results.append({"id": row["id"], "position": position, **metrics})
                if image is None and position == len(frames) - 1:
                    image = Image.new("RGB", (640, 256))
                    image.paste(Image.fromarray(pair[1]), (0, 0))
                    image.paste(Image.fromarray(np.rint(reconstructed * 255).astype(np.uint8)), (320, 0))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output.with_suffix(".png"))
    summary = {}
    for label, selected in (("early", results[::2]), ("terminal", results[1::2])):
        summary[label] = {key: float(np.mean([r[key] for r in selected]))
                          for key in ("mae_255", "psnr", "horizontal_gradient_ratio")}
    output.write_text(json.dumps({"codec": args.codec, "manifest": args.manifest,
                                  "cases": len(rows), "summary": summary,
                                  "per_frame": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
