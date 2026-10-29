#!/usr/bin/env python3
"""Select a validated codec and prepare isolated 10k cached SFT runs."""

import argparse
import json
from pathlib import Path

from futureworlds.checkpoints import file_sha256, verify_bundle

def load(path):
    return json.loads(Path(path).read_text())


def refresh_resume(config_path, output):
    config = load(config_path)
    checkpoints = sorted((output / "checkpoints").glob("step_*/experiment.json"))
    if checkpoints:
        config["resume"] = str(checkpoints[-1].parent)
    else:
        config.pop("resume", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", choices=("success", "natural"), default="success")
    parser.add_argument("--min-psnr-gain", type=float, default=1.0)
    parser.add_argument("--min-gradient-ratio", type=float, default=0.8)
    args = parser.parse_args()
    root = Path(args.root)
    codec_run = root / "runs/paper_adapt_success_codec"
    base = root / "assets/futureworlds_bundle_rt1_init"
    bundle = root / "assets/futureworlds_bundle_robocasa_codec_10k"
    complete_path = codec_run / "complete.json"
    if not complete_path.is_file() or load(complete_path).get("state") != "COMPLETE":
        raise ValueError(f"Codec training is not complete: {complete_path}")
    baseline = load(codec_run / "baseline.json")
    metrics = [json.loads(line) for line in (codec_run / "metrics.jsonl").read_text().splitlines()]
    candidates = [row for row in metrics if "validation" in row and
                  (codec_run / "checkpoints" / f"step_{row['step']:06d}" / "experiment.json").is_file()]
    if not candidates:
        raise ValueError("No validated codec checkpoint")
    best = max(candidates, key=lambda row: row["validation"]["psnr_from_mean_mse"])
    checkpoint = codec_run / "checkpoints" / f"step_{best['step']:06d}"
    receipt = load(checkpoint / "experiment.json")
    if any(file_sha256(checkpoint / name) != sha for name, sha in receipt["files"].items()):
        raise ValueError("Codec checkpoint checksum mismatch")
    validation = best["validation"]
    gain = validation["psnr_from_mean_mse"] - baseline["psnr_from_mean_mse"]
    if gain < args.min_psnr_gain or validation["horizontal_gradient_ratio"] < args.min_gradient_ratio:
        raise ValueError(f"Codec quality gate failed: PSNR gain {gain:.3f} dB, "
                         f"gradient ratio {validation['horizontal_gradient_ratio']:.3f}")
    if file_sha256(checkpoint / "diffusion_pytorch_model.safetensors") == file_sha256(base / "robocasa/codec/diffusion_pytorch_model.safetensors"):
        raise ValueError("Adapted codec is identical to the old codec")
    verify_bundle(base)
    if not bundle.exists():
        from install_robocasa_codec_bundle import install_bundle
        install_bundle(base, checkpoint, bundle)
    verify_bundle(bundle)
    if load(bundle / "manifest.json").get("robocasa_codec_sha256") != file_sha256(
            checkpoint / "diffusion_pytorch_model.safetensors"):
        raise ValueError("Existing adapted bundle uses another codec checkpoint")

    names = {
        "success": ("success_only_sft_cached_batched", "success_only_sft_new_codec_10k",
                    "success_new_codec_10k", "train_success.json"),
        "natural": ("natural_sft_cached_batched", "natural_sft_new_codec_10k",
                    "natural_new_codec_10k", "train_natural.json"),
    }
    original_name, run_name, cache_name, manifest_name = names[args.variant]
    original = root / "runs" / f"{original_name}.json"
    config_path = root / "runs" / f"{run_name}.json"
    output = root / "runs" / run_name
    cache = root / "cache/token_cache" / cache_name
    config = load(original)
    assert config["steps"] == 10000 and config["global_batch"] == 64 and config["micro_batch"] == 8
    assert config["train_manifest"] == str(root / "robocasa_data" / manifest_name)
    config.update(bundle=str(bundle), token_cache=str(cache), output=str(output))
    if output.exists() or cache.exists() or config_path.exists():
        if not config_path.is_file():
            raise ValueError("Partially prepared experiment; inspect existing paths")
        existing = load(config_path)
        existing.pop("resume", None)
        if existing != config:
            raise ValueError("Existing SFT config differs from the selected experiment")
        refresh_resume(config_path, output)
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        (root / "runs" / f"{run_name}_selection.json").write_text(json.dumps({
            "codec_checkpoint": str(checkpoint), "codec_step": best["step"],
            "baseline": baseline, "selected_validation": validation,
            "psnr_gain_db": gain, "sft_config": str(config_path),
        }, indent=2) + "\n")
    print(json.dumps({"bundle": str(bundle), "config": str(config_path),
                      "cache": str(cache), "output": str(output),
                      "codec_checkpoint": str(checkpoint), "psnr_gain_db": gain,
                      "variant": args.variant}))


if __name__ == "__main__":
    main()
