"""Batch independent episodes without mixing beam groups or reward baselines."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import torch

from .data import read_case
from .jev_judge import atomic_json
from .memory import make_prefix


def pack_condition(conditions, expansion=1):
    if all(not c for c in conditions):
        return {}
    if not all(c for c in conditions):
        raise ValueError("Cannot mix text-enabled and text-disabled episodes")
    width = max(c["text_features"].shape[1] for c in conditions)
    features, masks = [], []
    for c in conditions:
        f, m = c["text_features"], c["text_mask"]
        if f.shape[0] != 1 or m.shape != f.shape[:2]:
            raise ValueError("Expected one instruction per window")
        padded = f.new_zeros(1, width, f.shape[2])
        mask = m.new_zeros(1, width)
        padded[:, :f.shape[1]] = f
        mask[:, :m.shape[1]] = m
        features.append(padded)
        masks.append(mask)
    return dict(text_features=torch.cat(features).repeat_interleave(expansion, 0),
                text_mask=torch.cat(masks).repeat_interleave(expansion, 0))


def bind_condition(model, condition):
    if hasattr(model, "set_text"):
        model.set_text(condition.get("text_features"), condition.get("text_mask"))


@torch.no_grad()
def rollout_windows(model, context, anchor, actions, horizon, penalty,
                    conditions, groups=4, beams=2, visual=4375, width=80):
    """Window-major [window, group, beam] paths; parent selection is local."""
    model.eval()
    windows = context.shape[0]
    if anchor.shape[0] != windows or actions.shape[0] != windows:
        raise ValueError("Window dimensions do not match")
    paths = groups * beams
    count = windows * paths
    device = context.device
    expanded_context = context.repeat_interleave(paths, 0)
    expanded_actions = actions.transpose(0, 1).repeat_interleave(paths, 1)
    bind_condition(model, pack_condition(conditions, paths))
    history = {1: anchor.repeat_interleave(paths, 0)}
    scores = torch.full((windows, paths), -torch.inf, device=device, dtype=torch.float64)
    scores[:, ::beams] = 0
    raw = torch.zeros(count, device=device, dtype=torch.float64)
    frames = []
    window_offsets = (torch.arange(windows, device=device) * paths)[:, None]
    for target in range(2, horizon + 2):
        original, kept = make_prefix(expanded_context, history, expanded_actions, target)
        ancestors = torch.arange(count, device=device)
        codes = torch.empty(count, 0, device=device, dtype=torch.long)
        logs = torch.empty(count, 0, device=device)
        pending, cache = original, None
        for token in range(width):
            output = model(input_ids=pending, past_key_values=cache, use_cache=True)
            lp = output.logits[:, -1, :visual].float().log_softmax(-1).reshape(windows, paths, visual)
            parents, choices, next_scores = [], [], []
            for g in range(groups):
                start = g * beams
                frequency = torch.zeros(windows, visual, device=device, dtype=torch.float64)
                if g:
                    earlier = torch.cat(choices, 1)
                    frequency.scatter_add_(1, earlier, torch.ones_like(earlier, dtype=torch.float64))
                    frequency /= g * beams
                totals = scores[:, start:start + beams, None] + lp[:, start:start + beams].double() - penalty * frequency[:, None, :]
                score, flat = totals.flatten(1).topk(beams, dim=1, sorted=True)
                parents.append(start + flat // visual)
                choices.append(flat % visual)
                next_scores.append(score)
            parent = (torch.cat(parents, 1) + window_offsets).flatten()
            chosen = torch.cat(choices, 1).flatten()
            if not torch.equal(parent // beams, torch.arange(count, device=device) // beams):
                raise ValueError("A beam crossed its window or diversity group")
            flat_lp = lp.reshape(count, visual)
            raw = raw[parent] + flat_lp[parent, chosen].double()
            scores = torch.cat(next_scores, 1)
            codes = torch.cat([codes[parent], chosen[:, None]], 1)
            logs = torch.cat([logs[parent], flat_lp[parent, chosen, None]], 1)
            ancestors = ancestors[parent]
            cache = output.past_key_values
            cache.reorder_cache(parent)
            pending = chosen[:, None]
        for f in frames:
            for key in ("prefix", "response", "model_logp"):
                f[key] = f[key][ancestors]
        history = {t: x[ancestors] for t, x in history.items()}
        history[target] = codes
        frames.append(dict(target=target, prefix=original[ancestors], response=codes, model_logp=logs))
    selected = torch.arange(0, count, beams, device=device)
    frames = [{k: v[selected] if torch.is_tensor(v) else v for k, v in f.items()} for f in frames]
    history = {t: x[selected] for t, x in history.items()}
    for f in frames:
        rebuilt, _ = make_prefix(context.repeat_interleave(groups, 0), history,
            actions.transpose(0, 1).repeat_interleave(groups, 1), f["target"])
        if not torch.equal(rebuilt, f["prefix"]):
            raise ValueError("A returned candidate has an inconsistent episode history")
    rebuilt = sum(f["model_logp"].double().sum(-1) for f in frames)
    if not torch.allclose(rebuilt, raw[selected], atol=1e-7, rtol=0):
        raise ValueError("Raw proposal scores disagree with selected histories")
    return frames


def train_microbatch(pipe, actor, forward, reference, reward_fn, cfg, index,
                     schedule, sampler, step, micro, windows, accum, seed):
    from .training import policy_backward
    device = next(actor.parameters()).device
    cuda = device.type == "cuda"
    out = Path(cfg["output"])
    started = time.monotonic()
    times = {}
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    cursors = [(step - 1) * cfg["global_batch"] + micro * windows + i for i in range(windows)]
    cases = []
    for cursor in cursors:
        if schedule is None:
            row, _ = sampler.pick(cursor)
        else:
            row = index[schedule[cursor]["id"]]
        images, actions, text = read_case(cfg["train_manifest"], row, pipe.action_dim, cfg["horizon"])
        ctx, dynamics, acts, pixels, cond = pipe.encode_training_window(images, actions, text)
        cases.append(dict(ctx=ctx, anchor=dynamics[1:2], acts=acts, pixels=pixels, cond=cond, text=text))
    if cuda:
        torch.cuda.synchronize(device)
    times["encode_seconds"] = time.monotonic() - started

    def progress(phase):
        record = dict(step=step, microbatch=micro + 1, microbatches_per_step=accum,
                      windows=windows, cursors=cursors, phase=phase, timings=dict(times),
                      elapsed_seconds=time.monotonic() - started)
        if cuda:
            record.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        atomic_json(out / "progress.json", record)
        return record

    progress("rollout")
    before = time.monotonic()
    frames = rollout_windows(actor,
        torch.cat([c["ctx"].reshape(1, -1) + 4375 for c in cases]),
        torch.cat([c["anchor"] for c in cases]),
        torch.stack([c["acts"] for c in cases]), cfg["horizon"], cfg.get("diversity_penalty", .1),
        [c["cond"] for c in cases])
    if cuda:
        torch.cuda.synchronize(device)
    times["rollout_seconds"] = time.monotonic() - before
    progress("reward")

    def score(i):
        c = cases[i]
        selected = [{k: v[i * 4:(i + 1) * 4] if torch.is_tensor(v) else v
                     for k, v in f.items()} for f in frames]
        return reward_fn(selected, c["ctx"], c["acts"], c["pixels"],
                         task=c["text"], cursor=cursors[i], seed=seed)

    before = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=windows) as pool:
            scored = list(pool.map(score, range(windows)))
    except Exception as error:
        raise RuntimeError("Judge failed before optimizer update: " + str(error)) from error
    if cuda:
        torch.cuda.synchronize(device)
    times["reward_wall_seconds"] = time.monotonic() - before
    reward = torch.cat([value for value, stats in scored])
    informative = sum(int(float(value.std()) > 1e-6) for value, stats in scored)
    stats = {}
    for _, result in scored:
        for name, value in result.items():
            stats[name] = stats.get(name, 0.) + float(value)
    condition = pack_condition([c["cond"] for c in cases], 4)
    bind_condition(actor, condition)
    bind_condition(reference, condition)
    progress("backward")
    before = time.monotonic()
    loss = policy_backward(actor, forward, reference, frames, reward, condition,
        cfg["horizon"], cfg["clip"], cfg["kl_coef"], scale=1 / accum,
        group_ids=np.repeat(np.arange(windows), 4))
    if cuda:
        torch.cuda.synchronize(device)
    times["backward_seconds"] = time.monotonic() - before
    record = progress("microbatch_complete")
    with (out / "microbatch_metrics.jsonl").open("a") as stream:
        import json
        stream.write(json.dumps(record) + "\n")
    print("MICROBATCH", record, flush=True)
    return loss, stats, informative
