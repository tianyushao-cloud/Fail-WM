"""Cube-continuity artifacts in predictions vs GT, from the red-cube tracker on saved predictions.

Per case and model, over the 32 predicted frames:
  jumps      consecutive frames where the tracked cube moves > 25 px (GT never needs this)
  dup        frames with >1 red blob of >= 6 px (ghost / duplicated cube)
  vanish     frames where GT tracks a cube but the prediction has none
Usage: flicker.py SPLIT LABEL [LABEL ...]
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path("/media/tyshao/Jev-WM/experiments/sibling_action_eval_20261025")
sys.path.insert(0, "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/source/src")
from futureworlds.data import read_case
from futureworlds.robocasa_object_reward import track

DATA = "/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data"
split, labels = sys.argv[1], sys.argv[2:]
MANP = f"{DATA}/{split}.json"
rows = json.loads(Path(MANP).read_text())["cases"]


def stats(tr, gt_tr):
    xy = [t["xy"] for t in tr]
    jumps = sum(a is not None and b is not None and np.hypot(a[0] - b[0], a[1] - b[1]) > 25 for a, b in zip(xy, xy[1:]))
    dup = sum(t["count"] > 1 for t in tr)
    vanish = sum(g["xy"] is not None and t["xy"] is None for g, t in zip(gt_tr, tr))
    return np.array([jumps, dup, vanish])


tot = {l: [] for l in ["GT"] + labels}
for r in rows:
    images, _, _ = read_case(MANP, r, 52, 32)
    gt_tr = track(images[2:34])
    tot["GT"].append(stats(gt_tr, gt_tr))
    for l in labels:
        tot[l].append(stats(track(np.load(R / "results" / l / f"{r['id']}_pred.npz")["prediction"]), gt_tr))
print(f"{len(rows)} cases x 32 frames;  columns: total events / cases with >=1 event")
print(f"{'':16s} {'jump>25px':>16s} {'duplicate cube':>16s} {'vanished cube':>16s}")
for l, v in tot.items():
    v = np.array(v)
    print(f"{l:16s}" + "".join(f"{int(v[:, k].sum()):8d} / {int((v[:, k] > 0).sum()):3d}  " for k in range(3)))
