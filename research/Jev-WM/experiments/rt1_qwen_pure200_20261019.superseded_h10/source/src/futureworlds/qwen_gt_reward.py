"""Pure GT-referenced local Qwen-VL reward. Codec decoding only; no MAE/LPIPS reward."""
import time
from pathlib import Path

import numpy as np
import torch

from .batch import TensorBatch
from .jev_judge import atomic_json
from .qwen_judge import QwenJudge
from .rewards import CodecWorker


class QwenGTReward:
    def __init__(self, codec, config, device):
        self.worker = CodecWorker(codec, None)
        self.judge = QwenJudge(config["model"], device, scale=config.get("scale", 1),
                               batch_size=config.get("batch_size", 6))
        # Groups whose score spread does not exceed the calibrated presentation-
        # order noise are ties: zero advantage, never amplified by GRPO's std.
        self.tau = float(config["tie_tau"])
        self.root = Path(config["log_dir"])

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels, *, task, cursor, seed):
        started = time.monotonic()
        codes = torch.stack([f["response"] for f in frames], 1)
        if codes.shape[:2] != (4, 10) or codes.min() < 0 or codes.max() >= 4375:
            raise ValueError("Invalid RT-1 candidates")
        pred = self.worker.detokenize(TensorBatch(dict(tokens=codes, ctx_tokens=context.expand(4, -1, -1))), None).batch["pixels"][:, 1:]

        def rgb(x):
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite RGB")
            return x.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        # Future GT is introduced here only, after all four rollouts finished.
        initial = rgb(pixels[0, :2])
        reference = rgb(pixels[0, 2:12])
        predictions = np.stack([rgb(x) for x in pred])
        s = self.judge.score(initial, reference, predictions, task)
        scores = s["combined"]
        spread = float(scores.max() - scores.min())
        informative = spread > self.tau
        rewards = scores if informative else np.zeros_like(scores)
        atomic_json(self.root / f"{cursor:06d}.json", dict(
            cursor=cursor, seed=seed, task=str(task), scores=scores.tolist(), per_order=s["per_order"].tolist(),
            per_question=s["per_question"].tolist(), spread=spread, tau=self.tau, informative=bool(informative),
            frame_repeats=[int(sum(np.array_equal(a, b) for a, b in zip(p[:-1], p[1:]))) for p in predictions]))
        stats = dict(judged_groups=1, informative_groups=int(informative), score_sum=float(scores.sum()),
                     spread_sum=spread, judge_seconds=time.monotonic() - started)
        return torch.tensor(rewards, dtype=torch.float32, device=codes.device), stats
