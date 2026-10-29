"""Check the training reward module (scipy NCC, no cv2) against offline groups.pkl (cv2) and run the full __call__
with a stub codec worker that returns the logged predictions. Uses the training venv."""
import json, pickle, sys, time, types
import numpy as np, torch
sys.path.insert(0, "/media/tyshao/Jev-WM/experiments/robocasa_traj_memspo_g8_8gpu_20261027/source/src")
from futureworlds import robocasa_traj_reward as M

off_score = M.score  # identical formula; offline errors come from cv2 matching
V0 = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025"
G = {g["cursor"]: g for g in pickle.load(open("groups.pkl", "rb"))}
cfg = json.load(open("/media/tyshao/Jev-WM/experiments/robocasa_traj_memspo_g8_8gpu_20261027/train.json"))["traj_reward"]
cfg = dict(cfg, log_dir="/media/tyshao/Jev-WM/experiments/traj_reward_20261027/smoke", schedule=V0 + "/schedule.json")

class Stub:
    def __init__(self): self.pred = None
    def detokenize(self, batch, _):
        p = torch.from_numpy(self.pred).float().div(255).permute(0, 1, 4, 2, 3)
        p = torch.cat([p[:, :1], p], 1)  # module drops frame 0
        return types.SimpleNamespace(batch=dict(pixels=p))
R = M.RoboCasaTrajReward.__new__(M.RoboCasaTrajReward)
R.config = cfg; R.horizon = 32; R.root = __import__("pathlib").Path(cfg["log_dir"]); R.weights = cfg["weights"]
R.w = (.5, .25, .25); R.cap = 80.; R.vanish = 8.
R.schedule = json.load(open(cfg["schedule"])); R.cases = {r["id"]: r for r in json.load(open(cfg["train_manifest"]))["cases"]}
R.worker = Stub()
diffs, grip_diff, secs = [], [], []
for c in sorted(G)[::16]:
    z = np.load(f"{V0}/judge_log/{c:06d}_clips.npz")
    R.worker.pred = z["predictions"]
    pixels = torch.from_numpy(np.concatenate([z["initial"], z["reference"]])[None]).float().div(255).permute(0, 1, 4, 2, 3)
    frames = [dict(response=torch.from_numpy(z["tokens"][:, t])) for t in range(32)]
    t0 = time.monotonic()
    reward, stats = R(frames, torch.zeros(1, 2, 80, dtype=torch.long), None, pixels, task="t", cursor=c, seed=0)
    secs.append(time.monotonic() - t0)
    d = json.load(open(R.root / f"{c:06d}.json"))
    off = np.array([off_score(x["err"], 80, 8, (.5, .25, .25)) for x in G[c]["cands"]])
    diffs.append(np.abs(np.array(d["raw_reward"]) - off).max())
    grip_diff += [float(np.abs(np.array(e["grip"]) - x["err"]["grip"]).max()) for e, x in zip(d["errors"], G[c]["cands"])]
print(f"groups {len(diffs)}  max|reward new-offline| {max(diffs):.4f}  median {np.median(diffs):.5f}")
print(f"per-candidate max gripper shift diff px: median {np.median(grip_diff):.2f} p95 {np.percentile(grip_diff,95):.2f} max {max(grip_diff):.2f}")
print(f"seconds/group median {np.median(secs):.1f} max {max(secs):.1f}; stats keys {sorted(stats)}")
print("example", json.dumps({k: d[k] for k in ("raw_reward", "reward", "fake_lift_frames", "gt_lift")})[:400])
