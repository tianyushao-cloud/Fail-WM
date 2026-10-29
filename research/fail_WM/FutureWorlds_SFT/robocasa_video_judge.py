#!/usr/bin/env python3
"""RGB-only RoboCasa outcome judge and generated-video evaluator.

The judge uses only rendered RGB frames. It detects the red cube and green
goal, measures their image-plane separation over the last frames, and learns a
single success threshold from real validation videos. This makes the real-video
validation an explicit gate before generated predictions are scored.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def load_manifest(path):
    value = read_json(path)
    labels = {}
    for row in value["cases"]:
        if row["id"] in labels:
            raise ValueError(f"Duplicate id: {row['id']}")
        labels[row["id"]] = row
    if not labels:
        raise ValueError(f"Empty manifest: {path}")
    return value, labels


def centroid(mask):
    y, x = np.where(mask)
    if not len(x):
        return None
    return np.array([x.mean(), y.mean()], dtype=np.float64), int(len(x))


def frame_features(frame):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB frame, got {frame.shape} {frame.dtype}")
    x = frame.astype(np.float32) / 255.0
    spread = x.max(axis=2) - x.min(axis=2)
    red = (
        (x[:, :, 0] > 0.30)
        & (x[:, :, 0] > x[:, :, 1] * 1.35)
        & (x[:, :, 0] > x[:, :, 2] * 1.35)
        & (spread > 0.12)
    )
    green = (
        (x[:, :, 1] > 0.25)
        & (x[:, :, 1] > x[:, :, 0] * 1.18)
        & (x[:, :, 1] > x[:, :, 2] * 1.08)
        & (spread > 0.08)
    )
    rc = centroid(red)
    gc = centroid(green)
    if rc is None or gc is None:
        return {"distance": None, "red_area": rc[1] if rc else 0, "green_area": gc[1] if gc else 0}
    return {
        "distance": float(np.linalg.norm(rc[0] - gc[0])),
        "red_area": rc[1],
        "green_area": gc[1],
    }


def video_features(frames, tail=5):
    if frames.ndim != 4 or len(frames) < 1:
        raise ValueError(f"Expected [T,H,W,3] video, got {frames.shape}")
    values = [frame_features(frame) for frame in frames[-tail:]]
    distances = [v["distance"] for v in values if v["distance"] is not None]
    if not distances:
        return {
            "frames": int(len(frames)),
            "tail": int(min(tail, len(frames))),
            "observed": 0,
            "distance_median": None,
            "distance_mean": None,
            "distance_std": None,
            "red_area_median": float(np.median([v["red_area"] for v in values])),
            "green_area_median": float(np.median([v["green_area"] for v in values])),
        }
    return {
        "frames": int(len(frames)),
        "tail": int(min(tail, len(frames))),
        "observed": int(len(distances)),
        "distance_median": float(np.median(distances)),
        "distance_mean": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "red_area_median": float(np.median([v["red_area"] for v in values])),
        "green_area_median": float(np.median([v["green_area"] for v in values])),
    }


def real_video(path, horizon):
    with np.load(path, allow_pickle=False) as z:
        frames = z["images"].copy()
    # The first two frames are observations consumed by the world model. A
    # non-positive horizon means the complete audited future trajectory.
    return frames[2:] if horizon <= 0 else frames[2 : 2 + horizon]


def prediction_video(path, horizon):
    with np.load(path, allow_pickle=False) as z:
        frames = z["prediction"].copy()
    if frames.dtype != np.uint8:
        if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError(f"Prediction pixels must be uint8 or [0,1] float: {path}")
        frames = np.rint(frames * 255).astype(np.uint8)
    return frames if horizon <= 0 else frames[:horizon]


def collect_real(manifest_path, horizon, tail):
    manifest, labels = load_manifest(manifest_path)
    root = Path(manifest_path).parent
    rows = []
    for index, row in enumerate(manifest["cases"]):
        frames = real_video(root / row["file"], horizon)
        features = video_features(frames, tail)
        rows.append({
            "id": row["id"],
            "success": bool(row["success"]),
            "failure_type": row.get("failure_type") or "unknown_failure",
            "source": "real",
            "index": index,
            **features,
        })
    return rows


def collect_predictions(manifest_path, prediction_root, horizon, tail):
    manifest, _ = load_manifest(manifest_path)
    root = Path(prediction_root)
    rows = []
    for index, row in enumerate(manifest["cases"]):
        path = root / f"case_{index:06d}" / "prediction.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        features = video_features(prediction_video(path, horizon), tail)
        rows.append({
            "id": row["id"],
            "success": bool(row["success"]),
            "failure_type": row.get("failure_type") or "unknown_failure",
            "source": "prediction",
            "index": index,
            **features,
        })
    return rows


def confusion(rows, threshold):
    tp = fp = tn = fn = unavailable = unavailable_failure = 0
    for row in rows:
        score = row["distance_median"]
        if score is None:
            unavailable += 1
            unavailable_failure += not row["success"]
            continue
        predicted_success = score < threshold
        if not row["success"] and not predicted_success:
            tp += 1
        elif not row["success"] and predicted_success:
            fn += 1
        elif row["success"] and predicted_success:
            tn += 1
        else:
            fp += 1
    positive = tp + fn
    negative = tn + fp
    covered = positive + negative
    total_failure = positive + unavailable_failure
    sensitivity = tp / positive if positive else None
    specificity = tn / negative if negative else None
    balanced = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    return {
        "tp_failure": tp,
        "fn_failure_predicted_success": fn,
        "tn_success": tn,
        "fp_success_predicted_failure": fp,
        "unavailable": unavailable,
        "unavailable_failure": unavailable_failure,
        "coverage": covered / len(rows),
        "accuracy_on_covered": (tp + tn) / covered if covered else None,
        "accuracy_counting_unavailable_wrong": (tp + tn) / len(rows),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "optimism_rate": fn / positive if positive else None,
        "optimism_rate_upper_bound_if_unavailable_are_success": (
            (fn + unavailable_failure) / total_failure if total_failure else None
        ),
    }


def calibrate(rows):
    values = sorted({r["distance_median"] for r in rows if r["distance_median"] is not None})
    if not values:
        raise ValueError("No RGB cases have both cube and target detections")
    candidates = [values[0] - 1e-9]
    candidates.extend((a + b) / 2 for a, b in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-9)
    best = None
    for threshold in candidates:
        stats = confusion(rows, threshold)
        key = (
            -1.0 if stats["balanced_accuracy"] is None else stats["balanced_accuracy"],
            -stats["unavailable"],
            -stats["fn_failure_predicted_success"],
            -threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, stats)
    return {"threshold_pixels": float(best[1]), **best[2]}


def summarize(rows, threshold):
    stats = confusion(rows, threshold)
    groups = {"all": rows, "success": [r for r in rows if r["success"]], "failure": [r for r in rows if not r["success"]]}
    types = sorted({r["failure_type"] for r in rows if not r["success"]})
    groups.update({f"failure:{t}": [r for r in rows if not r["success"] and r["failure_type"] == t] for t in types})
    grouped = {}
    for name, values in groups.items():
        scores = [r["distance_median"] for r in values if r["distance_median"] is not None]
        grouped[name] = {
            "n": len(values),
            "observed": len(scores),
            "mean_distance_pixels": float(np.mean(scores)) if scores else None,
            "median_distance_pixels": float(np.median(scores)) if scores else None,
        }
    return {"n": len(rows), "groups": grouped, "classification": stats, "per_case": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--prediction-val")
    parser.add_argument("--prediction-test")
    parser.add_argument("--horizon", type=int, default=0, help="0 uses the complete trajectory")
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.horizon < 0 or args.tail < 1:
        raise ValueError("horizon must be non-negative and tail must be positive")

    real_val = collect_real(args.val_manifest, args.horizon, args.tail)
    real_test = collect_real(args.test_manifest, args.horizon, args.tail)
    calibration = calibrate(real_val)
    report = {
        "state": "COMPLETE",
        "judge": "red_cube_green_goal_image_distance_v1",
        "horizon": args.horizon,
        "tail": args.tail,
        "calibration_source": "real_validation_only",
        "calibration": calibration,
        "real_validation": summarize(real_val, calibration["threshold_pixels"]),
        "real_test": summarize(real_test, calibration["threshold_pixels"]),
    }
    if args.prediction_val:
        pred_val = collect_predictions(args.val_manifest, args.prediction_val, args.horizon, args.tail)
        report["prediction_validation"] = summarize(pred_val, calibration["threshold_pixels"])
    if args.prediction_test:
        pred_test = collect_predictions(args.test_manifest, args.prediction_test, args.horizon, args.tail)
        report["prediction_test"] = summarize(pred_test, calibration["threshold_pixels"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "calibration": calibration, "real_test": report["real_test"]["classification"]}, indent=2))


if __name__ == "__main__":
    main()
