"""Rule (red-cube tracker) + per-frame Qwen state QA checked against the h5 lift flag.

State question v6 (experiments/state_reward_v2_20261027): 4-way answer that separates
"on the table but the gripper is touching / closed on it" from "lifted", plus a yellow box on
Image 2 at the cube's start position (tracked in the last context frame, so identical for every
candidate). On simulator-labelled GT frames the previous 3-way question called grasped-on-table
frames lifted 98% of the time; v6: 0-8% false lift, 0.78 lifted recall, AUC 0.967 (was 0.921).

Offline benchmark (judge_benchmark_20261024): within-group ranking accuracy 0.93
on 382 contrast pairs, vs 0.65 for the pointwise interaction judge. Qwen never
sees the reference video: it labels each predicted frame as on-table / lifted /
missing, and the label timeline is compared with the simulator's lift flag for
the same packed frames (reference frame k = h5 index (2+k)*4).
"""
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from .batch import TensorBatch
from .rewards import CodecWorker
from .qwen_judge import QwenJudge, to_image
from PIL import ImageDraw
from .robocasa_object_reward import rule_scores, track
from .data import write_json

# Exact wording validated offline; small edits changed the answer distribution a lot.
STATE_Q = ("Two cropped frames from a robot simulation camera showing a gripper and a small RED CUBE. "
    "Image 1 is the start of the episode, with the cube resting on the table. Image 2 is a later frame. "
    "The yellow box in Image 2 marks where the cube was in Image 1. "
    "Question: in Image 2, what is the state of the red cube?\n"
    "0 = resting on the table, the gripper is not touching it.\n"
    "1 = still resting on the table, but the gripper is touching it, closed around it, or closed right next to it. "
    "The bottom of the cube is still on the table surface.\n"
    "2 = lifted: the cube is held up in the air, clearly above the table surface and higher than in Image 1.\n"
    "3 = no red cube is visible.\n"
    "A closed gripper near the cube does not mean the cube is lifted. "
    "Answer 2 only if the cube itself is visibly off the table. "
    "Answer with a single digit and nothing else.")
LIFTED, ANSWERS = 2, 4


def start_box(image, start, crop, scale, half=9):
    """Yellow outline at the cube's start position (full-frame start -> scaled crop coordinates)."""
    xy = track(start[None])[0]["xy"]
    if xy is None:
        return image
    x, y = (xy[0] - crop[0]) * scale, (xy[1] - crop[1]) * scale
    image = image.copy()
    ImageDraw.Draw(image).rectangle([x - half * scale, y - half * scale, x + half * scale, y + half * scale],
                                    outline=(255, 230, 0), width=2)
    return image


def state_items(start, predictions, ts, crop, scale):
    """Judge messages for every (candidate, timestamp); start = last context frame, full size."""
    x0, y0, x1, y1 = crop
    first = to_image(np.ascontiguousarray(start[y0:y1, x0:x1]), scale)
    items = []
    for p in predictions:
        for t in ts:
            frame = start_box(to_image(np.ascontiguousarray(p[t, y0:y1, x0:x1]), scale), start, crop, scale)
            items.append([dict(role="user", content=[
                dict(type="image", image=first), dict(type="image", image=frame),
                dict(type="text", text=STATE_Q)])])
    return items


class RoboCasaStateReward:
    def __init__(self, codec, config, device, horizon):
        self.worker = CodecWorker(codec, None)
        self.config = config
        self.horizon = horizon
        self.root = Path(config["log_dir"])
        self.judge = QwenJudge(config["model"], device, scale=2, batch_size=config.get("batch_size", 8))
        self.weights = config["weights"]
        if set(self.weights) != {"position", "persistence", "state"}:
            raise ValueError("weights must be position/persistence/state")
        self.schedule = json.loads(Path(config["schedule"]).read_text())
        self.cases = {r["id"]: r for r in json.loads(Path(config["train_manifest"]).read_text())["cases"]}
        self.ts = sorted(set(range(0, horizon, 2)) | {horizon - 1})

    def lift_flags(self, cursor):
        row = self.cases[self.schedule[cursor]["id"]]
        with h5py.File(row["source_h5"], "r") as f:
            flags = f["events/flags"][:, 2]
        index = (np.arange(self.horizon) + 2) * row.get("pack", 4)
        if index[-1] >= len(flags):
            raise ValueError(f"{row['id']}: h5 shorter than window")
        return flags[index].astype(np.float64), row["id"]

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels, *, task, cursor, seed):
        begin = time.monotonic()
        codes = torch.stack([f["response"] for f in frames], 1)
        n = codes.shape[0]
        if codes.shape[1] != self.horizon:
            raise ValueError("Unexpected horizon")
        pred = self.worker.detokenize(
            TensorBatch(dict(tokens=codes, ctx_tokens=context.expand(n, -1, -1))), None
        ).batch["pixels"][:, 1:]

        def rgb(x):
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite decoded pixels")
            return x.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

        initial = rgb(pixels[0, :2])
        reference = rgb(pixels[0, 2:2 + self.horizon])
        predictions = np.stack([rgb(x) for x in pred])
        del pred
        scored = [rule_scores(reference, p) for p in predictions]
        axes = {k: np.asarray([s[0][k] for s in scored]) for k in ("position", "persistence")}

        lift, case_id = self.lift_flags(cursor)
        items = state_items(initial[-1], predictions, self.ts, self.config["crop"], self.judge.scale)
        logits = self.judge.level_logits(items)[:, :ANSWERS].reshape(n, len(self.ts), ANSWERS)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        p_lift = probs[..., LIFTED]
        axes["state"] = 1 - np.abs(p_lift - lift[self.ts][None]).mean(-1)

        reward = sum(float(self.weights[k]) * axes[k] for k in self.weights)
        raw = reward.copy()
        if np.ptp(reward) < self.config.get("minimum_group_spread", 0.02):
            reward[:] = reward.mean()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.config.get("save_clips", True):
            np.savez_compressed(self.root / f"{cursor:06d}_clips.npz", initial=initial, reference=reference,
                                predictions=predictions, tokens=codes.cpu().numpy())
        write_json(self.root / f"{cursor:06d}.json", dict(
            cursor=cursor, seed=seed, id=case_id, task=str(task), weights=self.weights,
            axes={k: v.tolist() for k, v in axes.items()}, gt_lift=lift.tolist(), timestamps=self.ts,
            p_lift=p_lift.tolist(), p_contact=probs[..., 1].tolist(), p_missing=probs[..., 3].tolist(), raw_reward=raw.tolist(),
            reward=reward.tolist(), tracks=[s[1] for s in scored], seconds=time.monotonic() - begin))
        stats = dict(judged_groups=1, informative_groups=int(np.ptp(reward) > 1e-6),
                     judge_seconds=time.monotonic() - begin, reward_spread=float(np.ptp(raw)),
                     gt_lift_windows=int(lift.any()),
                     **{f"score_sum_{k}": float(v.mean()) for k, v in axes.items()})
        return torch.tensor(reward, dtype=torch.float32, device=codes.device), stats
