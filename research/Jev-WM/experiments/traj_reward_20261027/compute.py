"""Per-candidate trajectory errors + logged rewards + artifact stats on the v0 8-GPU run judge log."""
import json, glob, pickle, sys
from multiprocessing import Pool
import numpy as np, h5py
sys.path.insert(0, "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/source/src")
from futureworlds.robocasa_object_reward import track
from traj import gt_geometry, errors
RUN = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log"
MAN = {r["id"]: r for r in json.load(open("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"))["cases"]}
IDX = (np.arange(32) + 2) * 4


def artifacts(tr, gt_tr):
    xy = [t["xy"] for t in tr]
    jumps = sum(a is not None and b is not None and np.hypot(a[0] - b[0], a[1] - b[1]) > 25 for a, b in zip(xy, xy[1:]))
    dup = sum(t["count"] > 1 for t in tr)
    vanish = sum(g["xy"] is not None and t["xy"] is None for g, t in zip(gt_tr, tr))
    return dict(jumps=int(jumps), dup=int(dup), vanish=int(vanish))


def one(fn):
    d = json.load(open(fn)); row = MAN[d["id"]]; z = np.load(fn.replace(".json", "_clips.npz"))
    ref, preds = z["reference"], z["predictions"]
    with h5py.File(row["source_h5"]) as h:
        obj = h["observations/object_pose_wxyz"][:, :3][IDX]; eef = h["observations/eef_executed13"][:, :3][IDX]
        fl = h["events/flags"][:][IDX]
    geo = gt_geometry(obj, eef, ref)
    gt_tr = track(ref); y0 = next(g["xy"][1] for g in gt_tr if g["xy"] is not None)
    cands = []
    for k, p in enumerate(preds):
        tr = track(p)
        ys = np.array([np.nan if t["xy"] is None else t["xy"][1] for t in tr])
        cands.append(dict(err=errors(geo, ref, p), tl=np.where(np.isnan(ys), 0, (y0 - ys) > 8).astype(int),
                          **artifacts(tr, gt_tr), position=d["axes"]["position"][k], persistence=d["axes"]["persistence"][k],
                          state=d["axes"]["state"][k], v0=d["raw_reward"][k]))
    return dict(cursor=d["cursor"], id=d["id"], success=row["success"], failure_type=row["failure_type"],
                gt_lift=fl[:, 2].astype(int), grasp=fl[:, 1].astype(int), dz=(obj[:, 2] - obj[0, 2]) * 100, cands=cands)


if __name__ == "__main__":
    files = sorted(glob.glob(RUN + "/0*[0-9].json"))
    with Pool(48) as pool:
        out = pool.map(one, files, chunksize=2)
    pickle.dump(out, open("groups.pkl", "wb"))
    print(len(out), "groups")
