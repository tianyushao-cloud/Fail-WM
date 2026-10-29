#!/usr/bin/env python3
"""Screen real paired action branches against held-out ground truth."""

import json
from pathlib import Path

import numpy as np
import torch

from futureworlds.data import read_case, read_manifest
from futureworlds.pipeline import WorldPipeline, encode_actions


ROOT = Path("/media/tyshao/fail_WM/FutureWorlds_SFT")
OUT = ROOT / "evals_action_controlled"
MANIFEST = ROOT / "robocasa_data/test.json"
TEXT = "/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base"
PAIRS = ((30, 33), (40, 46), (80, 83))
HORIZON = 30


def center(frames):
    rgb = np.rint(frames * 255).astype(np.float32) if frames.dtype != np.uint8 else frames.astype(np.float32)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    mask = (red > 75) & (red > green * 1.5) & (red > blue * 1.4) & (green < 160)
    locations = []
    for m in mask:
        y, x = np.where(m)
        locations.append([float(x.mean()), float(y.mean())] if len(x) >= 8 else None)
    return locations


def metrics(pred, truth):
    p, t = center(pred), center(truth)
    err = [float(np.linalg.norm(np.array(a) - np.array(b))) if a and b else None
           for a, b in zip(p, t)]
    return {
        "cube_center_prediction": p,
        "cube_center_truth": t,
        "cube_center_error_px": err,
        "cube_center_error_at_frame25_px": err[23],
        "cube_center_error_at_frame30_px": err[28],
        "mean_cube_center_error_px": float(np.mean([v for v in err if v is not None])) if any(v is not None for v in err) else None,
        "full_frame_mae_255": float(np.mean(np.abs(pred * 255 - truth.astype(np.float32)))),
    }


def main():
    rows = read_manifest(MANIFEST, "robocasa")["cases"]
    report = {"horizon": HORIZON, "pairs": []}
    for a, b in PAIRS:
        ra, rb = rows[a], rows[b]
        ia, aa, ta = read_case(MANIFEST, ra, 52)
        ib, ab, tb = read_case(MANIFEST, rb, 52)
        if ra["initial_state_sha256"] != rb["initial_state_sha256"] or ta != tb:
            raise ValueError("Pair does not share initial state and instruction")
        d = float(np.mean(np.abs(ia[:2].astype(float) - ib[:2].astype(float))))
        if d > 0.01:
            raise ValueError("Initial views differ")
        report["pairs"].append({"indices": [a, b], "ids": [ra["id"], rb["id"]],
                                "success": [ra["success"], rb["success"]],
                                "first_two_image_mae_255": d, "models": {}})

    for model in ("success", "natural"):
        pipe = WorldPipeline(OUT / f"{model}_bundle", "robocasa", "sft", "cuda", TEXT)
        for pair in report["pairs"]:
            a, b = pair["indices"]
            ra, rb = rows[a], rows[b]
            ia, aa, ta = read_case(MANIFEST, ra, 52)
            ib, ab, tb = read_case(MANIFEST, rb, 52)
            tok_a = encode_actions(torch.as_tensor(aa, device=pipe.device), pipe.ranges, 52).cpu().numpy()
            tok_b = encode_actions(torch.as_tensor(ab, device=pipe.device), pipe.ranges, 52).cpu().numpy()
            changed = np.where((tok_a[:HORIZON + 1] != tok_b[:HORIZON + 1]).any(axis=1))[0]
            entry = {"first_different_action_row": int(changed[0]),
                     "changed_action_tokens": int(np.count_nonzero(tok_a[:HORIZON + 1] != tok_b[:HORIZON + 1]))}
            for name, actions, images in (("success_branch", aa, ia), ("failure_branch", ab, ib)):
                torch.manual_seed(0)
                with torch.inference_mode():
                    pred = pipe.predict(ia[:2], actions, ta, HORIZON, 4, "full")["prediction"]
                entry[name] = metrics(pred, images[2:2 + HORIZON])
                print(model, pair["ids"], name,
                      entry[name]["cube_center_error_at_frame25_px"], flush=True)
            pair["models"][model] = entry
    OUT.mkdir(exist_ok=True)
    (OUT / "screening.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote", OUT / "screening.json", flush=True)


if __name__ == "__main__":
    main()
