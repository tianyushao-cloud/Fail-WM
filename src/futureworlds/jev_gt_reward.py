"""Pure GT-conditioned judge reward. Codec decoding only; no MAE/LPIPS reward."""
import torch
import torch.distributed as dist
from .batch import TensorBatch
from .rewards import CodecWorker
from .jev_judge import Store


def synchronize_failure(error, world):
    """Every rank reaches this before policy backward, including failed judges."""
    message = None if error is None else f"{type(error).__name__}: {error}"
    messages = [message]
    if world > 1:
        messages = [None] * world
        dist.all_gather_object(messages, message)
    if any(x is not None for x in messages):
        raise RuntimeError("Judge failed before optimizer update: " + repr(messages))


class JevGTReward:
    def __init__(self, codec, config):
        self.worker = CodecWorker(codec, None)
        self.store = Store(config)

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels, *, task, cursor, seed):
        codes = torch.stack([f["response"] for f in frames], 1)
        if codes.shape[:2] != (4, 10) or codes.min() < 0 or codes.max() >= 4375:
            raise ValueError("Invalid RT-1 candidates")
        pred = self.worker.detokenize(TensorBatch(dict(tokens=codes, ctx_tokens=context.expand(4,-1,-1))), None).batch["pixels"][:,1:]
        def rgb(x):
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite RGB")
            return x.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,2,3,1).cpu().numpy()
        # Future GT is introduced here only, after all four rollouts finished.
        initial = rgb(pixels[0,:2])
        reference = rgb(pixels[0,2:12])
        import numpy as np
        predictions = np.stack([rgb(x) for x in pred])
        rewards, stats = self.store.score_group(initial, reference, predictions, task, seed=seed, cursor=cursor)
        return torch.tensor(rewards, dtype=torch.float32, device=codes.device), stats
