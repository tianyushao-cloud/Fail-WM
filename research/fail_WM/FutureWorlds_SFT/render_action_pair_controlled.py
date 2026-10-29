#!/usr/bin/env python3
"""Render a real success/failure action pair and both model predictions."""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from futureworlds.data import read_case, read_manifest
from futureworlds.pipeline import WorldPipeline


ROOT = Path("/media/tyshao/fail_WM/FutureWorlds_SFT")
OUT = ROOT / "evals_action_controlled"
MANIFEST = ROOT / "robocasa_data/test.json"
TEXT = "/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base"
INDICES = (30, 33)
HORIZON = 30


def encode(frames, target):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "960x568", "-r", "5", "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        input=np.stack(frames).tobytes(), capture_output=True, check=True,
    )


def main():
    rows = read_manifest(MANIFEST, "robocasa")["cases"]
    cases = [read_case(MANIFEST, rows[i], 52) for i in INDICES]
    first = cases[0][0][:2]
    if rows[30]["initial_state_sha256"] != rows[33]["initial_state_sha256"]:
        raise ValueError("Initial states differ")
    if np.mean(np.abs(first.astype(float) - cases[1][0][:2].astype(float))) > 0.01:
        raise ValueError("Initial RGB differs")
    predictions = {}
    for name in ("success", "natural"):
        pipe = WorldPipeline(OUT / f"{name}_bundle", "robocasa", "sft", "cuda", TEXT)
        for i, (images, actions, instruction) in zip(INDICES, cases):
            torch.manual_seed(0)
            with torch.inference_mode():
                prediction = pipe.predict(first, actions, instruction, HORIZON, 4, "full")["prediction"]
            predictions[name, i] = np.rint(prediction * 255).clip(0, 255).astype(np.uint8)
            print(name, rows[i]["id"], flush=True)

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
    )
    frames = []
    for step in range(HORIZON):
        canvas = Image.new("RGB", (960, 568), "black")
        draw = ImageDraw.Draw(canvas)
        for row, (i, (images, _, _)) in enumerate(zip(INDICES, cases)):
            top = row * 284
            label = "SUCCESS_ACTIONS" if row == 0 else "GRASP_MISS_ACTIONS"
            sources = (images[step + 2], predictions["success", i][step],
                       predictions["natural", i][step])
            for col, (source, title) in enumerate(zip(sources, ("REAL", "SUCCESS_ONLY", "SUCCESS_PLUS_FAILURE"))):
                canvas.paste(Image.fromarray(source), (col * 320, top + 28))
                draw.text((col * 320 + 7, top + 5), f"{label} | {title}",
                          fill="white", font=font)
        frames.append(np.asarray(canvas))
    OUT.mkdir(exist_ok=True)
    target = OUT / "real_action_pair_g00019_b00_b03.mp4"
    encode(frames, target)
    Image.fromarray(frames[23]).save(OUT / "real_action_pair_g00019_b00_b03_frame25.png")
    Image.fromarray(frames[-1]).save(OUT / "real_action_pair_g00019_b00_b03_frame30.png")
    receipt = {
        "success_id": rows[30]["id"], "failure_id": rows[33]["id"],
        "shared_initial_state_sha256": rows[30]["initial_state_sha256"],
        "context_frames_from": rows[30]["id"],
        "horizon": HORIZON, "fps": 5,
        "first_different_action_row": 14,
        "screening": str(OUT / "screening.json"),
        "video": str(target),
    }
    (OUT / "real_action_pair_g00019_b00_b03.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
