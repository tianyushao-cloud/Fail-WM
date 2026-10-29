#!/usr/bin/env python3
"""Compare two RoboCasa world models under fixed action interventions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from futureworlds.data import read_case, read_manifest
from futureworlds.pipeline import WorldPipeline, encode_actions
from preview_uncached_robocasa import encode


ROOT = Path("/media/tyshao/fail_WM/FutureWorlds_SFT")
TEXT_MODEL = Path("/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base")
START, END = 8, 26
SEED = 20261009


def interventions(actions):
    rng = np.random.default_rng(SEED)
    offset = rng.choice([-1.0, 1.0], size=3) * rng.uniform(0.3, 0.65, 3)
    pose = actions.copy().reshape(-1, 4, 13)
    profile = 0.25 + 0.75 * np.sin(np.linspace(0, np.pi, END - START))
    pose[START:END, :, :3] = np.clip(
        pose[START:END, :, :3] + profile[:, None, None] * offset,
        -1.0,
        1.0,
    )
    gripper = actions.copy().reshape(-1, 4, 13)
    gripper[START:END, :, 6] *= -1
    return {"original": actions, "random_xyz": pose.reshape(actions.shape),
            "flip_gripper": gripper.reshape(actions.shape)}, offset


def compose(predictions, path):
    names = ("original", "random_xyz", "flip_gripper")
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
    )
    frames = []
    for step in range(len(predictions["original"])):
        canvas = Image.new("RGB", (960, 284), "black")
        draw = ImageDraw.Draw(canvas)
        for col, name in enumerate(names):
            rgb = np.rint(predictions[name][step] * 255).clip(0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb), (col * 320, 28))
            draw.text((col * 320 + 8, 5), name.upper(), fill="white", font=font)
        frames.append(np.asarray(canvas))
    # The helper is fixed at 320x256, so encode the wider comparison directly.
    import subprocess

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "960x284", "-r", "5", "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        input=np.stack(frames).tobytes(), capture_output=True, check=True,
    )
    Image.fromarray(frames[-1]).save(path.with_suffix(".png"))


def run_model(name, bundle, images, actions, instruction, out):
    pipe = WorldPipeline(bundle, "robocasa", "sft", "cuda", str(TEXT_MODEL))
    horizon = len(images) - 2
    tokenized = {
        kind: encode_actions(torch.from_numpy(a).to(pipe.device), pipe.ranges, 52).cpu().numpy()
        for kind, a in actions.items()
    }
    base_tokens = tokenized["original"]
    tokens_changed = {
        kind: int(np.count_nonzero(tokens != base_tokens))
        for kind, tokens in tokenized.items() if kind != "original"
    }
    if any(count == 0 for count in tokens_changed.values()):
        raise ValueError("An intervention did not change model input tokens")
    if any(np.any(tokens[:START] != base_tokens[:START]) or
           np.any(tokens[END:] != base_tokens[END:])
           for kind, tokens in tokenized.items() if kind != "original"):
        raise ValueError("Intervention changed actions outside the selected window")

    predictions = {}
    for kind, a in actions.items():
        torch.manual_seed(0)
        with torch.inference_mode():
            predictions[kind] = pipe.predict(images[:2], a, instruction, horizon, 4, "full")["prediction"]
        encode(predictions[kind], out / f"{name}_{kind}.mp4")
        print(f"{name} {kind}: {horizon} frames", flush=True)
    compose(predictions, out / f"comparison_{name}.mp4")

    metrics = {"model": name, "horizon": horizon, "action_rows": [START, END],
               "changed_action_tokens": tokens_changed, "comparisons": {}}
    original = predictions["original"]
    for kind in ("random_xyz", "flip_gripper"):
        delta = np.abs(predictions[kind] - original) * 255
        per_frame = delta.mean(axis=(1, 2, 3))
        if np.max(delta[:START - 1]) > 1e-4:
            raise ValueError("Prediction changed before the modified action was used")
        metrics["comparisons"][kind] = {
            "first_changed_prediction_frame_zero_based": next(
                (int(i) for i, v in enumerate(per_frame) if v > 1e-4), None
            ),
            "mean_mae_255_after_action_window": float(per_frame[START - 1:].mean()),
            "last_frame_mae_255": float(per_frame[-1]),
            "last_frame_pixels_over_10_255_fraction": float((delta[-1].max(axis=2) > 10).mean()),
            "per_frame_mae_255": [float(v) for v in per_frame],
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=85)
    parser.add_argument("--output", type=Path, default=ROOT / "evals_action_intervention_new_codec")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "robocasa_data/test.json"
    row = read_manifest(manifest, "robocasa")["cases"][args.index]
    images, original, instruction = read_case(manifest, row, 52)
    if not END < len(images) - 1:
        raise ValueError("Episode is too short for this intervention")
    actions, offset = interventions(original)
    metrics = {"episode_id": row["id"], "test_index": args.index,
               "original_outcome": {"success": row["success"],
                                    "failure_type": row["failure_type"]},
               "seed": SEED, "action_rows_end_exclusive": [START, END],
               "random_xyz_offset": [float(x) for x in offset],
               "note": "Counterfactual actions have no ground-truth future video.",
               "models": []}
    for name in ("success", "natural"):
        metrics["models"].append(run_model(
            name, ROOT / "evals_preview_new_codec" / f"{name}_bundle",
            images, actions, instruction, args.output,
        ))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
