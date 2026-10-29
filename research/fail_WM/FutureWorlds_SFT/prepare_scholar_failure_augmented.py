#!/usr/bin/env python3
"""Add distinct training failures and prepare a 70%-failure scholar SFT run."""

import argparse
import concurrent.futures
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from futureworlds.data import read_manifest
from prepare_robocasa_sft import convert_episode, sha256, write_manifest
from prepare_scholar_robocasa_sft import prepare_bundle


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.root.resolve()
    data = root / "robocasa_data"
    old_source = Path("/media/tyshao/fail_WM/RoboCasa/data/batch1000")
    new_source = Path("/media/tyshao/fail_WM/RoboCasa/data/batch1000_failure70")
    source_manifest = new_source / "manifest.jsonl"
    old_train = read_manifest(data / "train_natural.json", "robocasa")
    old_val = read_manifest(data / "val.json", "robocasa")
    old_test = read_manifest(data / "test.json", "robocasa")
    old_ids = {r["id"] for m in (old_train, old_val, old_test) for r in m["cases"]}
    eval_groups = {r["initial_state_id"] for m in (old_val, old_test) for r in m["cases"]}
    existing_train_groups = {r["initial_state_id"] for r in old_train["cases"]}
    source_rows = [json.loads(line) for line in source_manifest.read_text().splitlines() if line]
    extra = sorted(
        (r for r in source_rows if r["split"] == "train" and not r["success"]
         and r["episode_id"] not in old_ids),
        key=lambda r: r["episode_id"],
    )
    if len(extra) != 233 or len({r["episode_id"] for r in extra}) != len(extra):
        raise ValueError(f"Unexpected distinct failure count: {len(extra)}")
    if any(r["initial_state_id"] in eval_groups or
           r["initial_state_id"] not in existing_train_groups for r in extra):
        raise ValueError("Extra failures cross the fixed initial-state split")

    cases_dir = data / "cases_failure70_extra"
    cases_dir.mkdir(exist_ok=True)
    jobs = [(r, new_source, cases_dir,
             "Pick up the cube and place it at the goal.", i)
            for i, r in enumerate(extra)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(lambda_job, jobs))
    for case in cases:
        if case["success"] or case["split"] != "train":
            raise ValueError("Converted non-failure or non-training case")

    combined = sorted(old_train["cases"] + cases, key=lambda r: r["episode_id"])
    if len(combined) != 1033 or len({r["id"] for r in combined}) != len(combined):
        raise ValueError("Combined training manifest is incomplete or duplicated")
    source_hash = hashlib.sha256(
        (old_train["source_manifest_sha256"] + sha256(source_manifest)).encode()
    ).hexdigest()
    train_path = data / "train_scholar_failure_augmented.json"
    write_manifest(train_path, combined, source_hash, "scholar_failure_augmented")
    read_manifest(train_path, "robocasa")

    successes = [r for r in combined if r["success"]]
    failures = [r for r in combined if not r["success"]]
    if (len(successes), len(failures)) != (443, 590):
        raise ValueError("Unexpected success/failure composition")
    bundle = root / "assets/futureworlds_scholar_robocasa"
    prepare_bundle(bundle)
    name = "failure_augmented70_scholar_sft_10k"
    schedule_path = root / "runs" / f"{name}_schedule.json"
    reference = json.loads((root / "runs/natural_scholar_sft_10k.json").read_text())
    rng = random.Random(reference["seed"])
    schedule = []
    for _ in range(reference["steps"] * reference["global_batch"] // 10):
        kinds = [False] * 7 + [True] * 3
        rng.shuffle(kinds)
        for success in kinds:
            row = rng.choice(successes if success else failures)
            schedule.append({"id": row["id"], "target": rng.randrange(2, row["frames"])})
    if schedule_path.exists():
        if json.loads(schedule_path.read_text()) != schedule:
            raise ValueError("Existing sampling schedule differs")
    else:
        write_json(schedule_path, schedule)

    config = dict(reference)
    config.update(
        train_manifest=str(train_path),
        schedule=str(schedule_path),
        token_cache=str(root / "cache/token_cache" / name),
        output=str(root / "runs" / name),
    )
    config_path = root / "runs" / f"{name}.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        existing.pop("resume", None)
        if existing != config:
            raise ValueError("Existing SFT config differs")
    checkpoints = sorted((root / "runs" / name / "checkpoints").glob("step_*/experiment.json"))
    if checkpoints:
        config["resume"] = str(checkpoints[-1].parent)
    write_json(config_path, config)
    summary = {
        "old_source": str(old_source),
        "new_source": str(new_source),
        "new_source_sha256": sha256(source_manifest),
        "new_failures": len(extra),
        "new_failure_types": dict(Counter(r["failure_type"] for r in extra)),
        "training_successes": len(successes),
        "training_failures": len(failures),
        "scheduled_successes": len(schedule) * 3 // 10,
        "scheduled_failures": len(schedule) * 7 // 10,
        "validation_manifest": str(data / "val.json"),
        "test_manifest": str(data / "test.json"),
        "train_manifest": str(train_path),
        "schedule": str(schedule_path),
        "config": str(config_path),
    }
    write_json(root / "runs" / f"{name}_selection.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def lambda_job(job):
    return convert_episode(*job)


if __name__ == "__main__":
    main()
