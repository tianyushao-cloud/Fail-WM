"""Validate gripper template matching on GT-vs-GT pairs with known projected displacement."""
import json, glob, numpy as np, h5py
from traj import grip_shift
from cam import project
RUN = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log"
MAN = {r["id"]: r for r in json.load(open("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"))["cases"]}
res = {1: [], 2: [], 4: [], 8: []}
for fn in sorted(glob.glob(RUN + "/0*.json"))[::12]:
    d = json.load(open(fn)); ref = np.load(fn.replace(".json", "_clips.npz"))["reference"]
    with h5py.File(MAN[d["id"]]["source_h5"]) as h: e = h["observations/eef_executed13"][:, :3]
    uv = project(e[(np.arange(32) + 2) * 4])
    for D in res:
        for k in range(0, 32 - D, 3):
            est, sc = grip_shift(ref[k], ref[k + D], uv[k])
            true = uv[k + D] - uv[k]
            res[D].append((np.hypot(*true), np.hypot(*(est - true)), sc))
for D, r in res.items():
    r = np.array(r); big = r[:, 0] > 5
    print(f"lag {D}: n={len(r)} true |d| median {np.median(r[:,0]):.1f}px  est err median {np.median(r[:,1]):.1f} p90 {np.percentile(r[:,1],90):.1f}  "
          f"| on |d|>5: n={big.sum()} med {np.median(r[big,1]) if big.any() else 0:.1f}  ncc {np.median(r[:,2]):.2f}")
