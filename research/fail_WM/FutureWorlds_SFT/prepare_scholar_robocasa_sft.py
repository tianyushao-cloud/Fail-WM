#!/usr/bin/env python3
"""Verify the released RoboCasa bundle and prepare two isolated SFT runs."""

import argparse
import json
import shutil
from pathlib import Path

from futureworlds.checkpoints import file_sha256, verify_bundle


REPO = "easylearning/FutureWorlds"
REVISION = "64b9802cb328b66c8ea263f2a4acd795c28bb144"
CODEC_SHA256 = "4ea6133eae9d3deba4fc546ac7d0cb318363d0d79372b29a8e7d1a753023b77d"
SFT_SHA256 = "a86d465e8b5c1cc824238b010aed477ea61d6d8307102135b886b32ca065463c"
VARIANTS = {
    "success": ("success_only_sft_cached_batched", "train_success.json"),
    "natural": ("natural_sft_cached_batched", "train_natural.json"),
}


def read(path):
    return json.loads(path.read_text())


def prepare_bundle(bundle):
    upstream = bundle / "release_manifest.json"
    if not upstream.exists():
        source = bundle / "manifest.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, upstream)
    published = read(upstream)
    files = [item for item in published["files"] if item["path"].startswith("robocasa/")]
    if not files:
        raise ValueError("Release manifest has no RoboCasa files")
    filtered = {
        "schema_version": 1,
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_manifest_sha256": file_sha256(upstream),
        "files": files,
    }
    manifest = bundle / "manifest.json"
    if not manifest.is_file() or read(manifest) != filtered:
        temp = manifest.with_suffix(".json.tmp")
        temp.write_text(json.dumps(filtered, indent=2) + "\n")
        temp.replace(manifest)
    verify_bundle(bundle)
    checks = {
        "robocasa/codec/diffusion_pytorch_model.safetensors": CODEC_SHA256,
        "robocasa/models/sft/model.safetensors": SFT_SHA256,
    }
    for name, expected in checks.items():
        actual = next((item["sha256"] for item in files if item["path"] == name), None)
        if actual != expected:
            raise ValueError(f"Published weight hash does not match the audited inventory: {name}")
    return filtered


def prepare_run(root, bundle, variant):
    original_name, manifest_name = VARIANTS[variant]
    config = read(root / "runs" / f"{original_name}.json")
    run_name = f"{variant}_scholar_sft_10k"
    output = root / "runs" / run_name
    cache = root / "cache/token_cache" / run_name
    config_path = root / "runs" / f"{run_name}.json"
    if config["steps"] != 10000 or config["global_batch"] != 64 or config["micro_batch"] != 8:
        raise ValueError("Unexpected reference SFT schedule")
    if config["train_manifest"] != str(root / "robocasa_data" / manifest_name):
        raise ValueError("Unexpected reference training data")
    config.update(bundle=str(bundle), token_cache=str(cache), output=str(output))
    config.pop("initial_checkpoint", None)
    config.pop("initial_sha256", None)
    config.pop("initialize_text_adapters", None)
    if config_path.exists():
        existing = read(config_path)
        existing.pop("resume", None)
        if existing != config:
            raise ValueError(f"Existing SFT config changed: {config_path}")
    elif output.exists() or cache.exists():
        raise ValueError(f"Experiment exists without config: {run_name}")
    checkpoints = sorted((output / "checkpoints").glob("step_*/experiment.json"))
    if checkpoints:
        config["resume"] = str(checkpoints[-1].parent)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return {"variant": variant, "manifest": config["train_manifest"],
            "cache": str(cache), "output": str(output), "config": str(config_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    bundle = root / "assets/futureworlds_scholar_robocasa"
    release = prepare_bundle(bundle)
    runs = [prepare_run(root, bundle, variant) for variant in VARIANTS]
    print(json.dumps({"repo": REPO, "revision": REVISION,
                      "bundle": str(bundle), "files": len(release["files"]),
                      "runs": runs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
