"""Pure local Qwen-VL multi-axis reward. Codec decoding only; no MAE/LPIPS reward."""
import time
from pathlib import Path

import numpy as np
import torch

from .batch import TensorBatch
from .jev_judge import atomic_json
from .qwen_axes import combine, score_axes
from .qwen_judge import QwenJudge
from .rewards import CodecWorker


class QwenGTReward:
    def __init__(self, codec, config, device, horizon):
        self.worker = CodecWorker(codec, None)
        self.judge = QwenJudge(config["model"], device, scale=config.get("scale", 1),
                               batch_size=config.get("batch_size", 4))
        self.weights = {k: float(v) for k, v in config["weights"].items()}
        # Per-axis calibrated noise: an axis whose group spread does not exceed
        # it contributes zero, so GRPO never amplifies pure judge noise.
        self.taus = {k: float(config["taus"][k]) for k in self.weights}
        self.horizon = horizon
        self.root = Path(config["log_dir"])

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels, *, task, cursor, seed):
        started = time.monotonic()
        codes = torch.stack([f["response"] for f in frames], 1)
        if codes.shape[:2] != (4, self.horizon) or codes.min() < 0 or codes.max() >= 4375:
            raise ValueError("Invalid RT-1 candidates")
        pred = self.worker.detokenize(TensorBatch(dict(tokens=codes, ctx_tokens=context.expand(4, -1, -1))), None).batch["pixels"][:, 1:]

        def rgb(x):
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite RGB")
            return x.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        # Future GT is introduced here only, after all four rollouts finished.
        initial = rgb(pixels[0, :2])
        reference = rgb(pixels[0, 2:2 + self.horizon])
        predictions = np.stack([rgb(x) for x in pred])
        axes = score_axes(self.judge, initial, reference, predictions, task, axes=list(self.weights))
        scores = {a: v["score"] for a, v in axes.items()}
        reward, informative = combine(scores, self.weights, self.taus)
        atomic_json(self.root / f"{cursor:06d}.json", dict(
            cursor=cursor, seed=seed, task=str(task), weights=self.weights, taus=self.taus,
            axes={a: dict(score=v["score"].tolist(), views=v["views"].tolist()) for a, v in axes.items()},
            informative_axes=informative, reward=reward.tolist(),
            frame_repeats=[int(sum(np.array_equal(a, b) for a, b in zip(p[:-1], p[1:]))) for p in predictions]))
        stats = dict(judged_groups=1, informative_groups=int(bool(informative)),
                     judge_seconds=time.monotonic() - started,
                     **{f"informative_{a}": int(a in informative) for a in self.weights},
                     **{f"score_sum_{a}": float(s.sum()) for a, s in scores.items()})
        return torch.tensor(reward, dtype=torch.float32, device=codes.device), stats
