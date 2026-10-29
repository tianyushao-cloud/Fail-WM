"""Fixed-scale combination; normalize only once in the existing RL objective."""
import math
import torch


class HybridReward:
    def __init__(self, base, decision, base_scale=1.0, weight=0.25):
        if not math.isfinite(base_scale) or base_scale <= 0 or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Hybrid reward scales must be finite and positive")
        self.base, self.decision = base, decision
        self.base_scale, self.weight = base_scale, weight

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels, *, task, raw_actions):
        # Fail before expensive base scoring if review receipts are incomplete.
        decision, metrics = self.decision(frames, context, actions, pixels,
                                          task=task, raw_actions=raw_actions)
        base, base_metrics = self.base(frames, context, actions, pixels)
        if base.shape != decision.shape or not torch.isfinite(base).all() or not torch.isfinite(decision).all():
            raise ValueError("Invalid hybrid reward components")
        reward = base / self.base_scale + self.weight * decision
        if not torch.isfinite(reward).all():
            raise ValueError("Nonfinite hybrid reward")
        return reward, {**base_metrics, **metrics,
                        "hybrid/base_mean": float(base.mean()),
                        "hybrid/mean": float(reward.mean())}
