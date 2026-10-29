"""Actual single-GPU horizon32 candidate/reward/backward/optimizer smoke."""
import json,time,sys
from pathlib import Path
import torch
from futureworlds.training import train
from futureworlds.rewards import VideoReward
R=Path(__file__).resolve().parents[1]
D=Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.25)
cfg=json.loads((R/"train.json").read_text())
cfg.update(steps=1,global_batch=1,checkpoint_every=1,output=str(D/"train"))
cfg.pop("resume",None)
(D/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
observed=[]
original=VideoReward.__call__
def checked_reward(self,frames,context,actions,pixels):
    assert len(frames)==32
    assert [f["target"] for f in frames]==list(range(2,34))
    assert pixels.shape[1]==34
    reward,stats=original(self,frames,context,actions,pixels)
    observed.append(dict(frames=len(frames),last_target=frames[-1]["target"],
        candidates=len(reward),reward_std=float(reward.std()),
        reward_mean=float(reward.mean()),stats=stats))
    return reward,stats
VideoReward.__call__=checked_reward
t=time.monotonic()
train(D/"config.json")
assert observed and all(x["reward_std"]>1e-6 for x in observed)
state=torch.load(D/"train/checkpoints/step_000001/training_state.pt",map_location="cpu",weights_only=False)
assert all(x["lr"]==1e-6 for x in state["optimizer"]["param_groups"])
receipt=dict(state="PASS",horizon=32,initial_observations=2,source_start=0,
    actual_optimizer_updates=1,global_batch=1,world_size=1,
    reward_checks=observed,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
    peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,seconds=time.monotonic()-t,
    formal_run_started=False,scope="Single GPU functional/memory check; does not validate 8-GPU collective communication")
(D/"result.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2),flush=True)

