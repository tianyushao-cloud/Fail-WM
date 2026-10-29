#!/usr/bin/env python3
"""Generate one comparable full-length RoboCasa preview from a trained bundle."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from futureworlds.data import read_case, read_manifest
from futureworlds.pipeline import WorldPipeline


def encode(frames, output):
    rgb = np.rint(frames * 255).astype(np.uint8)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x256", "-r", "5",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(command, input=rgb.tobytes(), capture_output=True, check=True)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    rows = read_manifest(args.manifest, "robocasa")["cases"]
    row = rows[args.index]
    pipe = WorldPipeline(args.bundle, "robocasa", "sft", "cuda", args.text_model)
    images, actions, instruction = read_case(args.manifest, row, pipe.action_dim)
    horizon = int(row["frames"]) - 2
    with torch.inference_mode():
        prediction = pipe.predict(images[:2], actions, instruction, horizon, 4, "full")["prediction"]
    if prediction.shape != (horizon, 256, 320, 3):
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predicted_rgb = encode(prediction, output)
    truth = images[2:]
    if truth.shape != predicted_rgb.shape:
        raise ValueError("Prediction and truth lengths differ")
    panel = Image.new("RGB", (960, 256))
    panel.paste(Image.fromarray(truth[-1]), (0, 0))
    panel.paste(Image.fromarray(predicted_rgb[-1]), (320, 0))
    panel.paste(Image.fromarray(predicted_rgb[0]), (640, 0))
    panel.save(output.with_suffix(".png"))
    details = {
        "episode_id": row["id"], "frames": horizon, "fps": 5,
        "last_frame_mae_255": float(np.abs(predicted_rgb[-1].astype(float) - truth[-1]).mean()),
        "prediction_horizontal_gradient_255": float(np.abs(np.diff(predicted_rgb[-1].astype(float), axis=1)).mean()),
        "truth_horizontal_gradient_255": float(np.abs(np.diff(truth[-1].astype(float), axis=1)).mean()),
        "mp4": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details), flush=True)


if __name__ == "__main__":
    main()
