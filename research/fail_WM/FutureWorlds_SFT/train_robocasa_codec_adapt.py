#!/usr/bin/env python3
"""Adapt the FutureWorlds visual codec to local RoboCasa success videos."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from futureworlds.codec import CompressiveVQModelFSQ
from futureworlds.codec_training.robocasa.common import augment_clip, gradient_penalty, install_checkpointing
from futureworlds.codec_training.robocasa.discriminator import Discriminator
from futureworlds.codec_training.robocasa.sampling import is_generator_step, target_times


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path, success_only):
    manifest = json.loads(Path(path).read_text())
    rows = [r for r in manifest["cases"] if not success_only or r["success"]]
    if not rows:
        raise ValueError(f"No matching cases: {path}")
    return rows


def load_images(data_root, row):
    with np.load(data_root / row["file"], allow_pickle=False) as data:
        images = data["images"].copy()
    if images.dtype != np.uint8 or images.shape[1:] != (256, 320, 3):
        raise ValueError(f"Invalid image array: {row['id']}")
    return images


def sample_clip(data_root, row, cursor, seed, device):
    images = load_images(data_root, row)
    rng = random.Random(seed + cursor * 9973)
    start = rng.randrange(max(1, len(images) - 40))
    times = target_times(len(images) - start, seed + 1000003 + cursor)
    selected = images[[start] + [start + t for t in times]]
    pixels = augment_clip(torch.from_numpy(selected).permute(0, 3, 1, 2).float(),
                          seed + 3000001 + cursor).to(device)
    return pixels[:1], pixels[1:]


@torch.inference_mode()
def evaluate(model, data_root, rows, device):
    model.eval()
    local = torch.zeros(4, device=device, dtype=torch.float64)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    for row in rows[rank::world]:
        frames = load_images(data_root, row)
        for index in (1, len(frames) - 1):
            pair = torch.from_numpy(frames[[0, index]]).permute(0, 3, 1, 2).float().to(device) / 255
            context, dynamics = model.tokenize(pair[None].contiguous())
            pred = model.detokenize(context, dynamics)[0, 1].float().clamp(0, 1)
            truth = pair[1]
            local[0] += (pred - truth).abs().mean().double()
            local[1] += (pred - truth).square().mean().double()
            local[2] += (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean().double()
            local[3] += (truth[:, :, 1:] - truth[:, :, :-1]).abs().mean().double()
    if dist.is_initialized():
        dist.all_reduce(local)
    n = len(rows) * 2
    return {"mae_255": float(local[0] / n * 255),
            "psnr_from_mean_mse": float(-10 * torch.log10((local[1] / n).clamp_min(1e-12))),
            "horizontal_gradient_ratio": float(local[2] / local[3])}


def save_checkpoint(folder, step, model, discriminator, optimizer, doptimizer,
                    fingerprint, config, world):
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
           "numpy": np.random.get_state(), "python": random.getstate()}
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states = [rng]
    if rank == 0:
        tmp = folder.with_name(folder.name + ".partial")
        if tmp.exists() or folder.exists():
            raise FileExistsError(folder)
        model.save_pretrained(tmp)
        torch.save({"step": step, "generator_optimizer": optimizer.state_dict(),
                    "discriminator_optimizer": doptimizer.state_dict(),
                    "discriminator": discriminator.state_dict(), "rng": states,
                    "world": world, "fingerprint": fingerprint}, tmp / "training_state.pt")
        receipt = {"step": step, "fingerprint": fingerprint, "config": config,
                   "files": {p.name: digest(p) for p in tmp.iterdir() if p.is_file()}}
        (tmp / "experiment.json").write_text(json.dumps(receipt, indent=2) + "\n")
        tmp.replace(folder)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world != cfg["world_size"] and not args.smoke:
        raise ValueError(f"Expected {cfg['world_size']} GPU processes, got {world}")
    if cfg["global_batch"] % world:
        raise ValueError("Global batch must be divisible by world size")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    random.seed(cfg["seed"] + rank)
    train_manifest = Path(cfg["train_manifest"])
    val_manifest = Path(cfg["val_manifest"])
    train_rows = read_rows(train_manifest, True)
    val_rows = read_rows(val_manifest, True)
    if {r["initial_state_id"] for r in train_rows} & {r["initial_state_id"] for r in val_rows}:
        raise ValueError("Training and validation initial states overlap")
    if any(not r["success"] for r in train_rows):
        raise ValueError("Initial codec stage must contain successful trajectories only")
    out = Path(cfg["output"] + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({"config": cfg,
        "train_sha256": digest(train_manifest), "val_sha256": digest(val_manifest),
        "source_codec_sha256": digest(Path(cfg["source_codec"]) / "diffusion_pytorch_model.safetensors"),
        "world": world}, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        path = out / "run.json"
        if path.exists() and json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Existing run has another configuration")
        path.write_text(json.dumps({"fingerprint": fingerprint, "config": cfg}, indent=2) + "\n")

    model = CompressiveVQModelFSQ.from_pretrained(cfg["source_codec"], local_files_only=True).to(device)
    model.requires_grad_(True)
    install_checkpointing(model)
    discriminator = Discriminator(depth=6).to(device)
    perceptual = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0)
    doptimizer = torch.optim.AdamW(discriminator.parameters(), lr=cfg["lr"], weight_decay=0)
    step0 = 0
    checkpoints = sorted((out / "checkpoints").glob("step_*/experiment.json"))
    if checkpoints and not args.smoke:
        receipt = json.loads(checkpoints[-1].read_text())
        folder = checkpoints[-1].parent
        if receipt["fingerprint"] != fingerprint or any(digest(folder / name) != sha
                for name, sha in receipt["files"].items()):
            raise ValueError("Checkpoint validation failed")
        saved = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=False)
        if saved["world"] != world:
            raise ValueError("World size changed on resume")
        model.load_state_dict(CompressiveVQModelFSQ.from_pretrained(folder,
                              local_files_only=True).state_dict())
        discriminator.load_state_dict(saved["discriminator"])
        optimizer.load_state_dict(saved["generator_optimizer"])
        doptimizer.load_state_dict(saved["discriminator_optimizer"])
        step0 = saved["step"]
        state = saved["rng"][rank]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    disc = DDP(discriminator, device_ids=[local], broadcast_buffers=True) if world > 1 else discriminator
    initial = evaluate(model, val_manifest.parent, val_rows[:4] if args.smoke else val_rows, device) if step0 == 0 else None
    if rank == 0:
        print(json.dumps({"event": "initial", "step": step0, "metrics": initial,
                          "train_cases": len(train_rows), "val_cases": len(val_rows)}), flush=True)
        if initial is not None and not args.smoke:
            (out / "baseline.json").write_text(json.dumps(initial, indent=2) + "\n")
    total_steps = (2 if args.smoke else cfg["steps"])
    batch = (1 if args.smoke else cfg["global_batch"])
    disc_start = (1 if args.smoke else cfg["discriminator_start"])
    accumulation = batch // world
    for step in range(step0 + 1, total_steps + 1):
        generator = step == 1 if args.smoke else is_generator_step(step - 1, disc_start)
        g_updates = (1 if step >= 1 else 0) if args.smoke else min(step, disc_start) + max(0, step - disc_start + 1) // 2
        d_updates = step - g_updates
        active = optimizer if generator else doptimizer
        warm = min(1.0, (g_updates if generator else d_updates) / max(1, cfg["warmup"]))
        cosine = 0.5 * (1 + math.cos(math.pi * max(0, step - cfg["warmup"])
                                 / max(1, cfg["steps"] - cfg["warmup"])))
        for group in active.param_groups:
            group["lr"] = cfg["lr"] * warm * (cosine if generator else 1.0)
        active.zero_grad(set_to_none=True)
        model.train()
        discriminator.train().requires_grad_(not generator)
        totals = torch.zeros(5, device=device)
        for micro in range(accumulation):
            cursor = (step - 1) * batch + micro * world + rank
            row = train_rows[random.Random(cfg["seed"] + cursor * 7919).randrange(len(train_rows))]
            ref, truth = sample_clip(train_manifest.parent, row, cursor, cfg["seed"], device)
            sync = (ddp.no_sync() if world > 1 and micro < accumulation - 1
                    else contextlib.nullcontext())
            if generator:
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, pred_ref, commit, dyn_commit = ddp(
                        sample=ref, dyn_sample=truth, segment_len=7,
                        return_dict=False, return_loss=True)
                    pred, pred_ref = pred.float(), pred_ref.float()
                    l1 = (pred - truth).abs().mean() + (pred_ref - ref).abs().mean()
                    lp = perceptual(pred * 2 - 1, truth * 2 - 1).mean() + perceptual(
                        pred_ref * 2 - 1, ref * 2 - 1).mean()
                    adversarial = (-discriminator(torch.cat([pred_ref, pred])).mean()
                                   if step > disc_start else torch.zeros((), device=device))
                    loss = l1 + lp + commit + dyn_commit + cfg["gan_weight"] * adversarial
                    (loss / accumulation).backward()
                totals += torch.stack([loss, l1, lp, adversarial, torch.zeros_like(loss)]).detach() / accumulation
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, pred_ref, _, _ = model(sample=ref, dyn_sample=truth,
                        segment_len=7, return_dict=False, return_loss=True)
                truth = truth.detach().requires_grad_(True)
                d_sync = (disc.no_sync() if world > 1 and micro < accumulation - 1
                          else contextlib.nullcontext())
                with d_sync:
                    real, fake = disc(torch.cat([ref, truth, pred_ref.detach().float(),
                                                 pred.detach().float()])).chunk(2)
                    hinge = torch.relu(1 - real).mean() + torch.relu(1 + fake).mean()
                    gp = gradient_penalty(truth, real, cfg["gradient_penalty_weight"])
                    loss = hinge + gp
                    (loss / accumulation).backward()
                totals += torch.stack([loss, torch.zeros_like(loss), torch.zeros_like(loss),
                                       hinge, gp]).detach() / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters() if generator else discriminator.parameters(),
                                       1.0, error_if_nonfinite=True)
        active.step()
        if world > 1:
            dist.all_reduce(totals)
            totals /= world
        record = {"step": step, "branch": "generator" if generator else "discriminator",
                  "generator_updates": g_updates, "discriminator_updates": d_updates,
                  "loss": float(totals[0]), "l1": float(totals[1]), "lpips": float(totals[2]),
                  "adversarial_or_hinge": float(totals[3]), "gradient_penalty": float(totals[4])}
        if not args.smoke and (step % cfg["validate_every"] == 0 or step == total_steps):
            record["validation"] = evaluate(model, val_manifest.parent, val_rows, device)
        if rank == 0:
            with (out / ("smoke.jsonl" if args.smoke else "metrics.jsonl")).open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if args.smoke or step % 100 == 0 or "validation" in record:
                print(json.dumps(record), flush=True)
        if not args.smoke and (step % cfg["checkpoint_every"] == 0 or step == total_steps):
            save_checkpoint(out / "checkpoints" / f"step_{step:06d}", step, model,
                            discriminator, optimizer, doptimizer, fingerprint, cfg, world)
    if rank == 0 and not args.smoke:
        (out / "complete.json").write_text(json.dumps({
            "state": "COMPLETE", "steps": total_steps, "fingerprint": fingerprint,
        }, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
