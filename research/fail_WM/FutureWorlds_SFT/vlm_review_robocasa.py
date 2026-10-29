#!/usr/bin/env python3
"""VLM cross-check for selected RoboCasa generated rollouts.

This is a review tool, not the primary metric. It samples RGB frames from a
prediction and asks a local Qwen2.5-VL model for a structured task assessment.
Only borderline or unavailable cases need to be sent here.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT = """You are reviewing a robot video for one task: pick up the red cube and place it on the green target on the counter.
Inspect all frames in temporal order. Return JSON only with exactly these keys:
{"success": true|false|"uncertain", "cube_on_green_target": true|false|"uncertain", "gripper_released": true|false|"uncertain", "stable_at_end": true|false|"uncertain", "failure_type": "none"|"grasp_miss"|"contact_without_lift"|"dropped_before_goal"|"lifted_but_not_placed"|"object_off_counter"|"uncertain", "reason": "short factual phrase"}
Use success=true only when the cube is visibly on the green target, the gripper is away, and the final frames are stable. Do not infer success from the robot reaching toward the target. Use uncertain when the object or target is occluded."""


def parse_json(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"parse_error": "no_json", "raw": text}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "raw": text}
    return value


def sample_frames(path, count):
    with np.load(path, allow_pickle=False) as z:
        frames = z["prediction"].copy()
    if len(frames) == 0:
        raise ValueError(f"Empty prediction: {path}")
    if frames.dtype != np.uint8:
        if not np.issubdtype(frames.dtype, np.floating) or not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError(f"Expected uint8 or [0,1] float prediction: {path}")
        frames = np.rint(frames * 255).astype(np.uint8)
    ids = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    return [Image.fromarray(frames[i]) for i in ids], ids.tolist()


def review(model, processor, images, device):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=180, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer, parse_json(answer)


def select_cases(judge_json, section, threshold, margin, max_cases):
    if not judge_json:
        return None
    value = json.loads(Path(judge_json).read_text())
    if threshold is None:
        threshold = value.get("calibration", {}).get("threshold_pixels")
    cases = value.get(section, {}).get("per_case", [])
    selected = []
    for row in cases:
        distance = row.get("distance_median")
        if distance is None or threshold is None or abs(float(distance) - threshold) <= margin:
            selected.append(row["index"])
    return selected[:max_cases] if max_cases else selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-json")
    parser.add_argument("--judge-section", default="prediction_test")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    selected = select_cases(args.judge_json, args.judge_section, args.threshold, args.margin, args.max_cases)
    if selected is None:
        selected = list(range(min(args.max_cases, len(manifest["cases"]))))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    results = []
    for index in selected:
        row = manifest["cases"][index]
        path = Path(args.prediction_root) / f"case_{index:06d}" / "prediction.npz"
        images, frame_ids = sample_frames(path, args.frames)
        raw, parsed = review(model, processor, images, args.device)
        results.append({"index": index, "id": row["id"], "frame_ids": frame_ids, "vlm": parsed, "raw": raw})
        print(json.dumps({"index": index, "id": row["id"], "vlm": parsed}, ensure_ascii=False), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"state": "COMPLETE", "model": args.model, "prompt": PROMPT, "cases": results}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
