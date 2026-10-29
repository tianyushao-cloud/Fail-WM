#!/usr/bin/env python3
"""Convert the RoboCasa HDF5 episodes into FutureWorlds SFT manifests.

The collector stores 20 Hz RGB frames and 12-D controls. FutureWorlds' native
RoboCasa layout predicts one visual frame per four controls, so this script
keeps state indices 0, 4, 8, ... and packs each four-control block into four
13-D vectors (52 values). The last action row is always zero and is never used
as a control target.

Outcome fields are copied to the manifest for analysis only. They are not
written into the instruction string or the model input.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


IMAGE_SIZE = (320, 256)  # PIL uses (width, height); model expects H=256,W=320.
ACTION_DIM = 12
PACK = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.shape != (480, 640, 3):
        raise ValueError(f"unexpected RGB frame: {frame.dtype} {frame.shape}")
    return np.asarray(
        Image.fromarray(frame, mode="RGB").resize(
            IMAGE_SIZE, Image.Resampling.BICUBIC
        ),
        dtype=np.uint8,
    )


def convert_episode(
    row: dict, source_root: Path, cases_dir: Path, instruction: str, output_index: int
):
    episode_id = row["episode_id"]
    episode = source_root / "episodes" / episode_id
    h5_path = episode / "trajectory.h5"
    meta_path = episode / "metadata.json"
    if not h5_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing episode files for {episode_id}")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("episode_id") != episode_id:
        raise ValueError(f"metadata episode mismatch: {episode_id}")

    with h5py.File(h5_path, "r") as h5:
        rgb = h5["observations/rgb"]
        raw = np.asarray(h5["actions/raw"], dtype=np.float32)
        frames = int(rgb.shape[0])
        if rgb.shape[1:] != (480, 640, 3):
            raise ValueError(f"{episode_id}: unexpected RGB shape {rgb.shape}")
        if raw.ndim != 2 or raw.shape[1] != ACTION_DIM:
            raise ValueError(f"{episode_id}: unexpected action shape {raw.shape}")
        if frames != len(raw) + 1:
            raise ValueError(f"{episode_id}: frame/action alignment is broken")
        nblocks = len(raw) // PACK
        if nblocks < 2:
            raise ValueError(f"{episode_id}: too short after four-step packing")
        frame_ids = np.arange(nblocks + 1, dtype=np.int64) * PACK
        # HDF5 RGB is gzip-compressed. Read it once per episode; random access
        # to every fourth frame otherwise repeatedly inflates the same chunks.
        rgb_values = np.asarray(rgb)
        images = np.stack([resize_frame(rgb_values[int(i)]) for i in frame_ids], axis=0)
        actions = np.zeros((nblocks + 1, PACK * (ACTION_DIM + 1)), dtype=np.float32)
        blocks = raw[: nblocks * PACK].reshape(nblocks, PACK, ACTION_DIM)
        actions[:-1] = np.pad(blocks, ((0, 0), (0, 0), (0, 1))).reshape(
            nblocks, -1
        )

    target = cases_dir / f"{output_index:06d}.npz"
    np.savez_compressed(
        target,
        images=images,
        actions=actions,
        instruction=np.array(instruction),
    )
    return {
        "id": episode_id,
        "episode_id": episode_id,
        "file": str(target.relative_to(cases_dir.parent)),
        "sha256": sha256(target),
        "split": row["split"],
        "frames": int(len(images)),
        "task": metadata.get("task", row.get("task", "CounterCubePreview")),
        "initial_state_id": row.get("initial_state_id"),
        "initial_state_sha256": row.get("initial_state_sha256"),
        "success": bool(row.get("success", metadata.get("success", False))),
        "failure_type": row.get("failure_type", metadata.get("failure_type")),
        "source_h5": str(h5_path),
        "source_h5_sha256": sha256(h5_path),
        "pack": PACK,
        "resize": "Pillow-BICUBIC 480x640 -> 256x320",
    }


def write_manifest(path: Path, cases: list[dict], source_sha256: str, name: str):
    value = {
        "schema_version": 1,
        "dataset": "robocasa",
        "name": name,
        "source_manifest_sha256": source_sha256,
        "cases": cases,
        "action_alignment": "packed raw actions[t*4:(t+1)*4] -> frame t -> frame t+1",
        "action_layout": "four ordered 13-D vectors, flattened to 52-D",
        "instruction_policy": "constant task instruction; no success/failure labels",
        "image_policy": "Pillow BICUBIC direct resize; no crop/pad",
    }
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def stable_key(case: dict) -> str:
    return hashlib.sha256(case["episode_id"].encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--instruction",
        default="Pick up the cube and place it at the goal.",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases_dir = args.output / "cases"
    cases_dir.mkdir()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("empty source manifest")
    source_sha = sha256(args.manifest)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    jobs = [
        (row, args.source_root, cases_dir, args.instruction, i)
        for i, row in enumerate(rows)
    ]
    converted = [None] * len(jobs)
    if args.workers == 1:
        for i, job in enumerate(jobs, 1):
            converted[i - 1] = convert_episode(*job)
            if i % 25 == 0 or i == len(rows):
                print(f"converted {i}/{len(rows)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(convert_episode, *job): i for i, job in enumerate(jobs)}
            for done, i in ((f, pending[f]) for f in concurrent.futures.as_completed(pending)):
                converted[i] = done.result()
                finished = sum(x is not None for x in converted)
                if finished % 25 == 0 or finished == len(rows):
                    print(f"converted {finished}/{len(rows)}", flush=True)

    # Keep all outcome labels outside the NPZ model inputs, but expose them in
    # manifests so a fixed train/eval probe can stratify by failure type.
    train = [r for r in converted if r["split"] == "train"]
    val = [r for r in converted if r["split"] == "val"]
    test = [r for r in converted if r["split"] == "test"]
    success = [r for r in train if r["success"]]
    failure = [r for r in train if not r["success"]]
    n = min(len(success), len(failure))
    balanced = sorted(success, key=stable_key)[:n] + sorted(failure, key=stable_key)[:n]
    balanced.sort(key=lambda r: r["episode_id"])

    write_manifest(args.output / "train_natural.json", train, source_sha, "natural_mix")
    write_manifest(args.output / "train_success.json", success, source_sha, "success_only")
    write_manifest(args.output / "train_balanced.json", balanced, source_sha, "balanced_success_failure")
    write_manifest(args.output / "val.json", val, source_sha, "validation")
    write_manifest(args.output / "test.json", test, source_sha, "test")
    summary = {
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": source_sha,
        "episodes": len(converted),
        "train": len(train),
        "train_success": len(success),
        "train_failure": len(failure),
        "train_balanced": len(balanced),
        "val": len(val),
        "test": len(test),
        "instruction": args.instruction,
        "image_shape": [256, 320, 3],
        "action_shape_per_case": "[frames, 52]",
        "pack": PACK,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
