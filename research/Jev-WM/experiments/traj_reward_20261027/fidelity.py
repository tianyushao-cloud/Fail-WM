"""Within-group Spearman between each reward and ROI pixel fidelity (-MSE on the judge crop), plus gripper error by type."""
import pickle, collections, numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from traj import score
G = pickle.load(open("groups.pkl", "rb"))
RUN = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log"
def mse(c):
    z = np.load(f"{RUN}/{c:06d}_clips.npz"); r = z["reference"][:, 40:220, 130:310].astype(np.float32)
    return [float(((p[:, 40:220, 130:310].astype(np.float32) - r) ** 2).mean()) for p in z["predictions"]]
with Pool(48) as pool: M = pool.map(mse, [g["cursor"] for g in G])
S = {"rule": lambda c: .35 * c["position"] + .15 * c["persistence"], "VLMst": lambda c: c["state"], "v0tot": lambda c: c["v0"],
     "T_cube": lambda c: score(c["err"], 80, 8, (1, 0, 0)), "T_grip": lambda c: score(c["err"], 80, 8, (0, 1, 0)),
     "T_full": lambda c: score(c["err"], 80, 8, (.5, .25, .25)), "Tc+VLM": lambda c: .5 * score(c["err"], 80, 8, (1, 0, 0)) + .5 * c["state"]}
print("within-group Spearman(reward, -ROI MSE), mean over groups [GT-no-lift | GT-lift]")
for k, f in S.items():
    rs = [(spearmanr([f(c) for c in g["cands"]], -np.array(m))[0], g["gt_lift"].any()) for g, m in zip(G, M)]
    a = np.array([r for r, _ in rs if np.isfinite(r)]); n = np.array([r for r, l in rs if np.isfinite(r) and not l]); l = np.array([r for r, l in rs if np.isfinite(r) and l])
    print(f"  {k:8s} all {a.mean():+.3f}  [{n.mean():+.3f} | {l.mean():+.3f}]")
print("mean gripper shift px by failure type (frames 0-15 / 16-31)")
d = collections.defaultdict(list)
for g in G:
    for c in g["cands"]: d[g["failure_type"] or "success"].append((c["err"]["grip"][:16].mean(), c["err"]["grip"][16:].mean()))
for k, v in sorted(d.items()): v = np.array(v); print(f"  {k:24s} {v[:,0].mean():5.1f} / {v[:,1].mean():5.1f}")
