"""Optional cached batched SFT path.

This module deliberately lives beside the original trainer. The existing
``futureworlds train`` path remains unchanged; this entry point keeps the same
sample cursor, optimizer schedule, checkpoint format, and effective batch while
collating cached single-example inputs per rank.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP

from futureworlds.cache import load_cached
from futureworlds.checkpoints import file_sha256
from futureworlds.data import disjoint, read_manifest, write_json
from futureworlds.pipeline import WorldPipeline
from futureworlds.protocol import feature_fingerprint, implementation, text_identity
from futureworlds.sampling import Schedule
from futureworlds.training import save_checkpoint, sft_example
from action_dropout_20261012 import apply_action_dropout, source_identity


def _pad_examples(examples, pad_id):
    """Right-pad variable-length causal inputs without changing real positions."""
    max_len = max(item["input_ids"].shape[1] for item in examples)
    result = {}
    for key, fill in (
        ("input_ids", pad_id),
        ("labels", -100),
        ("attention_mask", 0),
        ("position_ids", 0),
    ):
        values = []
        for item in examples:
            value = item[key]
            width = max_len - value.shape[1]
            if width:
                value = F.pad(value, (0, width), value=fill)
            values.append(value)
        result[key] = torch.cat(values, dim=0)
    return result


def _collate_condition(conditions):
    if not conditions or not conditions[0]:
        return {}
    widths = {item["text_features"].shape[1] for item in conditions}
    if len(widths) != 1:
        raise ValueError("Cached text features have inconsistent widths")
    return {
        "text_features": torch.cat(
            [item["text_features"] for item in conditions], dim=0
        ),
        "text_mask": torch.cat([item["text_mask"] for item in conditions], dim=0),
    }


def _load_actor(cfg, pipe, device, method):
    actor = pipe.model
    if cfg.get("initial_checkpoint"):
        from transformers import AutoConfig, AutoModelForCausalLM
        from futureworlds.text_conditioning import TextWorldModel

        initial = Path(cfg["initial_checkpoint"])
        if file_sha256(initial / "model.safetensors") != cfg["initial_sha256"]:
            raise ValueError("Initial checkpoint hash mismatch")
        core = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(initial, local_files_only=True),
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        core.load_state_dict(load_file(str(initial / "model.safetensors")), strict=True)
        if pipe.settings["text_conditioning"]["enabled"]:
            tc = pipe.settings["text_conditioning"]
            actor = TextWorldModel(
                core, text_width=tc["width"], layers=tc["layers"], inner=tc["inner"]
            )
            adapter = initial / "text_adapters.safetensors"
            if adapter.exists():
                actor.adapters.load_state_dict(load_file(str(adapter)), strict=True)
            elif method != "sft" or not cfg.get("initialize_text_adapters", False):
                raise ValueError("Missing trained text adapters")
        else:
            actor = core
        actor = actor.to(device)
    return actor


def _select(row_index, schedule, sampler, cursor):
    if schedule is not None:
        entry = schedule[cursor]
        return row_index[entry["id"]], entry
    row, target = sampler.pick(cursor)
    return row, {"target": target}


def train(config_path):
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get("method") != "sft":
        raise ValueError("batched_sft only supports method=sft")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = cfg.get("device", "cuda")
    if device == "cuda":
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    if world > 1:
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")

    dropout = cfg["action_dropout"]
    if dropout != {"probability": 0.25, "null_token": 9006, "seed_offset": 60261012, "cursor_multiplier": 130363}:
        raise ValueError("Unexpected dropout protocol")
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    global_batch = int(cfg["global_batch"])
    micro_batch = int(cfg["micro_batch"])
    if micro_batch < 1 or global_batch != world * micro_batch:
        raise ValueError("global_batch must equal WORLD_SIZE * micro_batch")

    trainset = read_manifest(cfg["train_manifest"], cfg["dataset"])
    if any(row.get("split") != "train" for row in trainset["cases"]):
        raise ValueError("Training requires an explicitly designated train split")
    if cfg.get("evaluation_manifest"):
        disjoint(trainset, read_manifest(cfg["evaluation_manifest"], cfg["dataset"]))
    rows = trainset["cases"]
    row_index = {row["id"]: row for row in rows}
    schedule = None
    if cfg.get("schedule"):
        schedule = json.loads(Path(cfg["schedule"]).read_text())
        for entry in schedule:
            if entry["id"] not in row_index:
                raise ValueError("Schedule refers to missing training data")
        if len(schedule) < cfg["steps"] * global_batch:
            raise ValueError("Incomplete training schedule")
    elif cfg.get("paper_locked", False):
        raise ValueError("Paper-locked training requires the original schedule")

    cache_root = Path(cfg["token_cache"])
    if not (cache_root / "complete.json").is_file():
        raise FileNotFoundError(f"Incomplete token cache: {cache_root}")
    feature_hash = feature_fingerprint(
        cfg["bundle"], cfg["dataset"], cfg.get("text_model")
    )
    cache_meta = json.loads((cache_root / "complete.json").read_text())
    if cache_meta.get("manifest_sha256") != file_sha256(cfg["train_manifest"]):
        raise ValueError("Token cache manifest does not match training manifest")
    if cache_meta.get("bundle_sha256") != feature_hash:
        raise ValueError("Token cache feature fingerprint does not match config")
    if cache_meta.get("cases") != len(rows):
        raise ValueError("Token cache case count does not match training manifest")

    effective = {k: v for k, v in cfg.items() if k not in ("resume", "output")}
    effective.update(
        implementation=implementation(),
        trainer_variant="cached_batched_sft_adrop_v1",
        extension_sources=source_identity(),
        text_assets=text_identity(
            cfg["bundle"], cfg["dataset"], cfg.get("text_model")
        ),
        train_manifest_sha256=file_sha256(cfg["train_manifest"]),
        bundle_manifest_sha256=file_sha256(Path(cfg["bundle"]) / "manifest.json"),
        world_size=world,
        schedule_sha256=file_sha256(cfg["schedule"]) if schedule is not None else None,
    )
    fingerprint = hashlib.sha256(
        json.dumps(effective, sort_keys=True).encode()
    ).hexdigest()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if (out / "run.json").exists():
            old = json.loads((out / "run.json").read_text())
            if not cfg.get("resume") or old["fingerprint"] != fingerprint:
                raise ValueError("Use a fresh output or resume identical configuration")
        write_json(out / "run.json", dict(fingerprint=fingerprint, config=effective))
    if world > 1:
        dist.barrier()

    pipe = WorldPipeline(
        cfg["bundle"],
        cfg["dataset"],
        cfg.get("initial_variant", "sft"),
        device,
        cfg.get("text_model"),
    )
    actor = _load_actor(cfg, pipe, device, "sft")
    pipe.model = actor
    if hasattr(actor, "adapters"):
        groups = [
            dict(params=actor.core.parameters(), lr=cfg["lr"]),
            dict(params=actor.adapters.parameters(), lr=cfg["adapter_lr"]),
        ]
    else:
        groups = actor.parameters()
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=0)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    step0 = 0
    if cfg.get("resume"):
        folder = Path(cfg["resume"])
        receipt = json.loads((folder / "experiment.json").read_text())
        if any(
            file_sha256(folder / name) != digest
            for name, digest in receipt["files"].items()
        ):
            raise ValueError("Checkpoint checksum mismatch")
        state = torch.load(
            folder / "training_state.pt", map_location="cpu", weights_only=False
        )
        if state["fingerprint"] != fingerprint or state["world_size"] != world:
            raise ValueError("Resume protocol/world-size mismatch")
        (actor.core if hasattr(actor, "core") else actor).load_state_dict(
            load_file(str(folder / "model.safetensors")), strict=True
        )
        if hasattr(actor, "adapters"):
            actor.adapters.load_state_dict(
                load_file(str(folder / "text_adapters.safetensors")), strict=True
            )
        opt.load_state_dict(state["optimizer"])
        step0 = state["step"]
        saved = state["rng"][rank]
        torch.set_rng_state(saved["torch"])
        np.random.set_state(saved["numpy"])
        random.setstate(saved["python"])
        generator.set_state(saved["generator"])
        if saved["cuda"] is not None:
            torch.cuda.set_rng_state(saved["cuda"])

    forward = (
        DDP(
            actor,
            device_ids=[local] if device.startswith("cuda") else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if world > 1
        else actor
    )
    sampler = Schedule(rows, cfg["dataset"], seed)
    model_config = actor.core.config if hasattr(actor, "core") else actor.config
    pad_id = getattr(model_config, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)

    for step in range(step0 + 1, cfg["steps"] + 1):
        started = time.monotonic()
        opt.zero_grad(set_to_none=True)
        entries = []
        for local_index in range(micro_batch):
            cursor = (step - 1) * global_batch + local_index * world + rank
            row, entry = _select(row_index, schedule, sampler, cursor)
            entries.append((row, entry, cursor))

        dropout_counts = torch.zeros(4, dtype=torch.long, device=device)
        examples = []
        conditions = []
        for row, entry, cursor in entries:
            ctx, dynamics, acts, cond = load_cached(
                cache_root, row, feature_hash, device
            )
            target = entry.get(
                "target",
                int(
                    np.random.default_rng(seed + cursor * 9973).integers(
                        2, len(dynamics)
                    )
                ),
            )
            if not 2 <= target < len(dynamics):
                raise ValueError("Invalid RoboCasa SFT target")
            original = sft_example(ctx, dynamics, acts, target, anchor=1)
            example, dropped = apply_action_dropout(original, cursor, seed, dropout["probability"])
            group = 0 if row["success"] else 1
            dropout_counts[group] += 1
            dropout_counts[2 + group] += int(dropped)
            examples.append(example)
            conditions.append(cond)
        batch_input = _pad_examples(examples, pad_id)
        cond = _collate_condition(conditions)

        actor.train(True)
        warm = cfg.get("warmup_steps", 500)
        progress = max(0, step - warm) / max(1, cfg["steps"] - warm)
        scale = min(step / max(1, warm), 1.0) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for gi, group in enumerate(opt.param_groups):
            group["lr"] = (cfg["lr"] if gi == 0 else cfg["adapter_lr"]) * scale

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and cfg.get("sft_bf16", True),
        ):
            loss = forward(**batch_input, **cond, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite SFT loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), 1.0, error_if_nonfinite=True
        )
        opt.step()

        value = loss.detach()
        if world > 1:
            dist.all_reduce(value)
            value /= world
        if world > 1:
            dist.all_reduce(dropout_counts)
        if rank == 0:
            record = dict(
                step=step,
                loss=float(value),
                grad_norm=float(norm),
                seconds=time.monotonic() - started,
                method="sft",
                trainer_variant="cached_batched_sft_adrop_v1",
                action_dropout_counts=dict(zip(("success_samples", "failure_samples", "success_dropped", "failure_dropped"), dropout_counts.tolist())),
                micro_batch=micro_batch,
                global_batch=global_batch,
            )
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % cfg.get("checkpoint_every", 50) == 0 or step == cfg["steps"]:
            save_checkpoint(
                actor,
                opt,
                out / "checkpoints" / f"step_{step:06d}",
                step,
                cfg,
                fingerprint,
                world,
                rank,
                generator,
            )
    if rank == 0:
        write_json(
            out / "complete.json",
            dict(state="COMPLETE", steps=cfg["steps"], fingerprint=fingerprint),
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(prog="batched_sft_adrop_20261012.py")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
