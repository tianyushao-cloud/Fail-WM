"""Fit the 640x480 analytic preview-camera projection -> 320x256 packed frame mapping using cube tracker."""
import json, glob, sys
import numpy as np, h5py
sys.path.insert(0, "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/source/src")
from futureworlds.robocasa_object_reward import track
RUN = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log"
MAN = {r["id"]: r for r in json.load(open("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"))["cases"]}
pos = np.array([.9, -1.4, 1.9]); xa = np.array([.83205, .5547, 0]); ya = np.array([-.2691, .40365, .8745])
xa /= np.linalg.norm(xa); ya /= np.linalg.norm(ya); za = np.cross(xa, ya)
f = 240 / np.tan(np.deg2rad(24))
def proj(p):
    c = (p - pos) @ np.stack([xa, ya, za]).T
    return np.stack([320 + f * c[..., 0] / -c[..., 2], 240 - f * c[..., 1] / -c[..., 2]], -1)
A, B = [], []
for fn in sorted(glob.glob(RUN + "/0*.json"))[:60:3]:
    d = json.load(open(fn)); row = MAN[d["id"]]
    ref = np.load(fn.replace(".json", "_clips.npz"))["reference"]
    with h5py.File(row["source_h5"]) as h:
        obj = h["observations/object_pose_wxyz"][:, :3]
    idx = (np.arange(32) + 2) * 4
    tr = track(ref)
    for k, t in enumerate(tr):
        if t["xy"] is not None and idx[k] < len(obj):
            A.append(proj(obj[idx[k]])); B.append(t["xy"])
A, B = np.array(A), np.array(B)
for ax in (0, 1):
    M = np.c_[A[:, ax], np.ones(len(A))]; s, res, *_ = np.linalg.lstsq(M, B[:, ax], rcond=None)
    print("axis", ax, "scale/offset", s, "rms px", np.sqrt(np.mean((M @ s - B[:, ax]) ** 2)), "n", len(A))
