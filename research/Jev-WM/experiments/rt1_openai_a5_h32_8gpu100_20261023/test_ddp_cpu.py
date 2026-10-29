"""Eight CPU/Gloo ranks verify restored Adam state and 2x8 == batch16.

Synthetic linear model only; this is not an eight-GPU RL/NCCL smoke test.
"""
import copy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def batch(step):
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randn(16, 5, generator=g, dtype=torch.float64), torch.randn(16, 3, generator=g, dtype=torch.float64)


def serial(model, opt, step):
    x, y = batch(step)
    opt.zero_grad()
    (model(x) - y).square().mean().backward()
    opt.step()


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    torch.manual_seed(1729)
    original = torch.nn.Linear(5, 3).double()
    old_opt = torch.optim.AdamW(original.parameters(), lr=1e-3, weight_decay=0)
    for step in range(1, 11):
        serial(original, old_opt, step)
    model = copy.deepcopy(original)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    opt.load_state_dict(copy.deepcopy(old_opt.state_dict()))
    parallel = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=True)
    max_error = 0.
    for step in [11, 12]:
        x, y = batch(step)
        opt.zero_grad()
        for micro in range(2):
            cursor = micro * 8 + rank
            import contextlib
            with parallel.no_sync() if micro == 0 else contextlib.nullcontext():
                ((parallel(x[cursor:cursor+1]) - y[cursor:cursor+1]).square().mean() / 2).backward()
        opt.step()
        serial(original, old_opt, step)
        for actual, expected in zip(model.parameters(), original.parameters()):
            max_error = max(max_error, float((actual - expected).abs().max().detach()))
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual, expected in zip(opt.state.values(), old_opt.state.values()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual[key], expected[key], atol=1e-12, rtol=1e-12)
    if rank == 0:
        result = dict(kind="synthetic CPU Gloo; not GPU RL", world_size=8,
                      accumulation=2, global_batch=16, restored_step=10,
                      continued_steps=[11, 12], max_parameter_error=max_error, passed=True)
        Path(__file__).with_name("ddp_cpu_check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
