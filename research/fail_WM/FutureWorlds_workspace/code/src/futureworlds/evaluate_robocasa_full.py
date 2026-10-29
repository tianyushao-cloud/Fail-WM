#!/usr/bin/env python3
"""Generate full-length RoboCasa rollouts for outcome judging.

The standard evaluator uses fixed horizons for cross-paper frame metrics. This
companion evaluates each case to its audited terminal frame (frames - 2), so a
success/failure judge is applied to a generated video with the same temporal
extent as the real episode.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from futureworlds.checkpoints import file_sha256
from futureworlds.data import read_case, read_manifest, write_json
from futureworlds.evaluation import Metrics
from futureworlds.pipeline import WorldPipeline
from futureworlds.protocol import implementation, text_identity


def evaluate(bundle, manifest, output, device, metric_device, text_model, beam_width, memory):
    m = read_manifest(manifest, "robocasa")
    rows = m["cases"]
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if device.startswith("cuda") or metric_device.startswith("cuda"):
        torch.cuda.set_device(local)
        if device.startswith("cuda"):
            device = f"cuda:{local}"
        if metric_device.startswith("cuda"):
            metric_device = f"cuda:{local}"
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pipe = WorldPipeline(bundle, "robocasa", "sft", device, text_model)
    metrics = Metrics(metric_device)
    protocol = {
        "dataset": "robocasa",
        "variant": "sft",
        "manifest_sha256": file_sha256(manifest),
        "bundle_manifest_sha256": file_sha256(Path(bundle) / "manifest.json"),
        "beam_width": beam_width,
        "memory": memory,
        "generation_device": device.split(":")[0],
        "metric_device": metric_device.split(":")[0],
        "implementation": implementation(),
        "text_assets": text_identity(bundle, "robocasa", text_model),
        "precision": "float32",
        "tf32": False,
        "initial_observations": 2,
        "gt_refresh": False,
        "full_episode_horizon": True,
        "cases": [r["id"] for r in rows],
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for index in range(rank, len(rows), world):
        row = rows[index]
        dest = out / f"case_{index:06d}"
        receipt = dest / "result.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old.get("fingerprint") != fingerprint or old.get("id") != row["id"]:
                raise ValueError("Refusing to mix old full-rollout results")
            continue
        horizon = int(row["frames"]) - 2
        if horizon < 1:
            raise ValueError(f"Episode too short for full rollout: {row['id']}")
        images, actions, text = read_case(manifest, row, pipe.action_dim)
        result = pipe.predict(images[:2], actions, text, horizon, beam_width, memory)
        values = metrics(result["prediction"], images[2 : 2 + horizon], [horizon])
        dest.mkdir(parents=True, exist_ok=True)
        tmp = dest / "prediction.partial"
        with tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                prediction=result["prediction"],
                tokens=result["tokens"],
                truth=images[2 : 2 + horizon],
                observations=images[:2],
            )
        tmp.replace(dest / "prediction.npz")
        write_json(
            receipt,
            {
                "id": row["id"],
                "fingerprint": fingerprint,
                "horizon": horizon,
                "prediction_sha256": file_sha256(dest / "prediction.npz"),
                **values,
                "trace": result["trace"],
            },
        )
        print(json.dumps({"event": "case_complete", "index": index + 1, "total": len(rows), "horizon": horizon}), flush=True)
    if world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("gloo")
        dist.barrier()
    if rank == 0:
        cases = [json.loads((out / f"case_{i:06d}/result.json").read_text()) for i in range(len(rows))]
        if any(c["fingerprint"] != fingerprint for c in cases):
            raise ValueError("Mixed full-rollout protocol results")
        write_json(out / "summary.json", {"state": "COMPLETE", "protocol": protocol, "fingerprint": fingerprint, "per_case": cases})
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--memory", choices=["full", "recent", "minimal"], default="full")
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
