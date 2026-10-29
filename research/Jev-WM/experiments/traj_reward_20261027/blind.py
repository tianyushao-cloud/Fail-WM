"""Blind pairs where Tc80v8 and the lift label disagree; side randomised; key saved separately."""
import pickle, itertools, json, numpy as np
from PIL import Image, ImageDraw
from traj import score
G = pickle.load(open("groups.pkl", "rb"))
RUN = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log"
T = lambda c: score(c["err"], 80, 8, (1, 0, 0))
rng = np.random.default_rng(1); key = []
for ft in ("contact_without_lift", "object_off_counter"):
    pool = []
    for g in G:
        if g["failure_type"] != ft: continue
        ag = np.array([(c["tl"] == g["gt_lift"]).mean() for c in g["cands"]]); t = np.array([T(c) for c in g["cands"]])
        for i, j in itertools.combinations(range(8), 2):
            if abs(ag[i] - ag[j]) >= .25 and (ag[i] - ag[j]) * (t[i] - t[j]) < 0:
                pool.append((g, i, j, int(ag[j] > ag[i])))  # label-better index position
    seen = set(); pick = []
    for n in rng.permutation(len(pool)):
        if pool[n][0]["cursor"] not in seen: seen.add(pool[n][0]["cursor"]); pick.append(pool[n])
        if len(pick) == 6: break
    for g, i, j, lb in pick:
        z = np.load(f"{RUN}/{g['cursor']:06d}_clips.npz"); ts = [0, 5, 10, 15, 20, 25, 31]
        sides = [i, j] if rng.random() < .5 else [j, i]
        rows = []
        for name, clip in (("GT", z["reference"]), ("X", z["predictions"][sides[0]]), ("Y", z["predictions"][sides[1]])):
            row = np.concatenate([clip[t, 40:220, 130:310] for t in ts], 1)
            im = Image.fromarray(np.ascontiguousarray(row)); ImageDraw.Draw(im).text((3, 3), name, fill=(0, 0, 255)); rows.append(np.array(im))
        fn = f"blind_{len(key):02d}.png"; Image.fromarray(np.concatenate(rows, 0)).save(fn)
        label_better = [i, j][lb]
        key.append(dict(fn=fn, ft=ft, cursor=g["cursor"], X=sides[0], Y=sides[1],
                        label_says="X" if sides[0] == label_better else "Y", T_says="Y" if sides[0] == label_better else "X"))
json.dump(key, open("blind_key.json", "w"), indent=1)
print(len(key))
