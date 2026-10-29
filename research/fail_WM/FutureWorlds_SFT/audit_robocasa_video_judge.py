#!/usr/bin/env python3
"""Audit source labels and the RGB judge under simple image degradation."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from robocasa_video_judge import confusion, load_manifest, real_video, video_features


def check_source_labels(manifest_path):
    manifest, _ = load_manifest(manifest_path)
    mismatches = []
    max_goal_xy_error_difference = 0.0
    for row in manifest["cases"]:
        source = Path(row["source_h5"])
        meta = json.loads(source.with_name("metadata.json").read_text())
        with h5py.File(source) as h5:
            final_success = bool(h5["events/flags"][-1, -1])
            cube_xy = h5["observations/object_pose_wxyz"][-1, :2]
        goal_xy = np.asarray(meta["goal_position_world"][:2])
        xy_error = float(np.linalg.norm(cube_xy - goal_xy))
        max_goal_xy_error_difference = max(
            max_goal_xy_error_difference,
            abs(xy_error - meta["outcome_metrics"]["final_goal_xy_error_m"]),
        )
        if (bool(row["success"]) != final_success
                or bool(row["success"]) != bool(meta["success"])
                or row.get("failure_type") != meta.get("failure_type")
                or bool(row["success"]) != bool(meta["event_summary"]["success"])):
            mismatches.append(row["id"])
    return {
        "n": len(manifest["cases"]),
        "label_mismatches": mismatches,
        "max_goal_xy_error_difference_m": max_goal_xy_error_difference,
    }


def degrade(frames, kind):
    if kind == "original":
        return frames
    if kind == "dim_70pct":
        return np.rint(frames.astype(np.float32) * 0.7).astype(np.uint8)
    if kind == "gaussian_blur_sigma_1_5":
        return np.stack([cv2.GaussianBlur(frame, (0, 0), 1.5) for frame in frames])
    if kind == "jpeg_quality_50":
        result = []
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            result.append(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        return np.stack(result)
    raise ValueError(kind)


def audit_images(manifest_path, threshold, tail):
    manifest, _ = load_manifest(manifest_path)
    kinds = ["original", "dim_70pct", "gaussian_blur_sigma_1_5", "jpeg_quality_50"]
    results = {kind: [] for kind in kinds}
    for row in manifest["cases"]:
        frames = real_video(Path(manifest_path).parent / row["file"], 0)[-tail:]
        for kind in kinds:
            features = video_features(degrade(frames, kind), tail)
            results[kind].append({"id": row["id"], "success": bool(row["success"]), **features})
    report = {}
    baseline = {r["id"]: r["distance_median"] for r in results["original"]}
    for kind, rows in results.items():
        changed = [r["id"] for r in rows if (r["distance_median"] is None) != (baseline[r["id"]] is None)
                   or (r["distance_median"] is not None and baseline[r["id"]] is not None
                       and (r["distance_median"] < threshold) != (baseline[r["id"]] < threshold))]
        errors = [r["id"] for r in rows if r["distance_median"] is not None
                  and (r["distance_median"] < threshold) != r["success"]]
        report[kind] = {"classification": confusion(rows, threshold),
                        "changed_vs_original": changed, "misclassified": errors}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--judge-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    judge = json.loads(Path(args.judge_report).read_text())
    if judge["horizon"] != 0:
        raise ValueError("Judge report must use full trajectories")
    threshold = judge["calibration"]["threshold_pixels"]
    report = {
        "threshold_pixels": threshold,
        "source_labels": {
            "validation": check_source_labels(args.val_manifest),
            "test": check_source_labels(args.test_manifest),
        },
        "real_test_image_degradation": audit_images(args.test_manifest, threshold, judge["tail"]),
        "scope": "Real held-out videos only; these perturbations do not estimate generated-video accuracy.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "source_labels": report["source_labels"],
                      "real_test_image_degradation": {k: v["classification"]
                                                     for k, v in report["real_test_image_degradation"].items()}}, indent=2))


if __name__ == "__main__":
    main()
