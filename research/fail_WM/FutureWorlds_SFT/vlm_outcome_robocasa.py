#!/usr/bin/env python3
"""Qwen video outcome probe with explicit pixel conversion and resumable output."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT = """These frames show a robot attempting to place a RED cube on a GREEN circular target. Frames are in chronological order. The final image is a magnified view of the table in the last frame. Judge the visual state at the END. Answer with exactly one word:
SUCCESS: the red cube is clearly resting on the green target after the gripper moves away.
FAILURE: the red cube is clearly elsewhere, dropped, or still held by the gripper.
UNCERTAIN: you cannot clearly see both the cube and target in the last frames, or cannot tell whether it was released.
Do not assume success merely because the robot approaches the target. Output only SUCCESS, FAILURE, or UNCERTAIN."""


def load_frames(path, key):
    with np.load(path, allow_pickle=False) as z:
        frames = z[key].copy()
    if key == "images":
        frames = frames[2:]
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 1:
        raise ValueError(f"Invalid video: {path}")
    if frames.dtype != np.uint8:
        if not np.issubdtype(frames.dtype, np.floating) or not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError(f"Expected uint8 or [0,1] float RGB video: {path}")
        frames = np.rint(frames * 255).astype(np.uint8)
    return frames


def select_frames(frames):
    n = len(frames)
    positions = [0, n // 3, 2 * n // 3, *range(max(0, n - 3), n)]
    ids = sorted(set(min(i, n - 1) for i in positions))
    images = [Image.fromarray(frames[i]) for i in ids]
    height, width = frames.shape[1:3]
    crop = images[-1].crop((int(width * 0.4), int(height * 0.12), width, int(height * 0.88)))
    images.append(crop.resize((width * 2, height * 2), Image.Resampling.BICUBIC))
    return images, ids


def classify(model, processor, images, device):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0].strip()
    value = raw.upper().strip(" .!\n")
    return value if value in {"SUCCESS", "FAILURE", "UNCERTAIN"} else "INVALID", raw


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", choices=["real", "prediction"], required=True)
    parser.add_argument("--prediction-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.source == "prediction" and not args.prediction_root:
        parser.error("--prediction-root is required for prediction source")
    manifest = json.loads(Path(args.manifest).read_text())
    cases = manifest["cases"]
    indices = list(range(len(cases))) if args.indices is None else args.indices
    if any(i < 0 or i >= len(cases) for i in indices):
        parser.error("case index outside manifest")
    protocol = {"model": str(Path(args.model).resolve()), "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "source": args.source, "prediction_root": str(Path(args.prediction_root).resolve()) if args.prediction_root else None,
                "prompt": PROMPT, "frame_policy": "start, one-third, two-thirds, final three, magnified final table crop", "pixel_policy": "uint8 or rounded [0,1] float"}
    output = Path(args.output)
    if output.exists():
        report = json.loads(output.read_text())
        if report["protocol"] != protocol:
            raise ValueError("Refusing to mix different VLM protocols")
    else:
        report = {"state": "PARTIAL", "protocol": protocol, "cases": []}
    done = {r["index"] for r in report["cases"]}
    pending = [i for i in indices if i not in done]
    if not pending:
        report["state"] = "COMPLETE" if len(done) == len(cases) else "PARTIAL"
        save(output, report)
        return
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=True, attn_implementation="sdpa").to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    for index in pending:
        row = cases[index]
        if args.source == "real":
            path = Path(args.manifest).parent / row["file"]
            key = "images"
        else:
            path = Path(args.prediction_root) / f"case_{index:06d}" / "prediction.npz"
            key = "prediction"
        images, frame_ids = select_frames(load_frames(path, key))
        label, raw = classify(model, processor, images, args.device)
        report["cases"].append({"index": index, "id": row["id"], "real_success": bool(row["success"]),
                                "frame_ids": frame_ids, "vlm_outcome": label, "raw": raw})
        report["state"] = "COMPLETE" if len(report["cases"]) == len(cases) else "PARTIAL"
        save(output, report)
        print(json.dumps({"index": index, "id": row["id"], "real_success": bool(row["success"]),
                          "vlm_outcome": label, "raw": raw}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
