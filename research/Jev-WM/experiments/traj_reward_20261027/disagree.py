"""Render pairs where T_cube and the lift-timeline label disagree (label-better = A)."""
import pickle, itertools, sys, json, numpy as np
from PIL import Image, ImageDraw
from traj import aggregate, CAP
G = pickle.load(open("groups.pkl", "rb"))
RUN = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log"
tc = lambda c: 1 - aggregate(c["err"]["cube"]) / CAP
rng = np.random.default_rng(0); out = []
for ft in ("contact_without_lift", "object_off_counter", "grasp_miss"):
    cand = []
    for g in G:
        if (g["failure_type"] or "success") != ft: continue
        ag = np.array([(c["tl"] == g["gt_lift"]).mean() for c in g["cands"]])
        for i, j in itertools.combinations(range(len(ag)), 2):
            if abs(ag[i] - ag[j]) >= .25:
                a, b = (i, j) if ag[i] > ag[j] else (j, i)
                if tc(g["cands"][a]) < tc(g["cands"][b]): cand.append((g, a, b))
    for g, a, b in [cand[i] for i in rng.choice(len(cand), min(4, len(cand)), replace=False)]:
        z = np.load(f"{RUN}/{g['cursor']:06d}_clips.npz"); ts = [0, 6, 12, 18, 24, 31]
        x0, y0, x1, y1 = 130, 40, 310, 220
        rows = []
        for name, clip in (("GT", z["reference"]), ("A", z["predictions"][a]), ("B", z["predictions"][b])):
            row = np.concatenate([clip[t, y0:y1, x0:x1] for t in ts], 1)
            im = Image.fromarray(np.ascontiguousarray(row)); ImageDraw.Draw(im).text((3, 3), name, fill=(0, 0, 255)); rows.append(np.array(im))
        fn = f"dis_{len(out):02d}.png"; Image.fromarray(np.concatenate(rows, 0)).save(fn)
        A, B = g["cands"][a], g["cands"][b]
        out.append(dict(fn=fn, ft=ft, cursor=g["cursor"], a=a, b=b, gt_lift_frames=int(g["gt_lift"].sum()),
                        tlA=int(A["tl"].sum()), tlB=int(B["tl"].sum()), TcA=round(tc(A), 3), TcB=round(tc(B), 3),
                        cubeA=A["err"]["cube"][ts].round(1).tolist(), cubeB=B["err"]["cube"][ts].round(1).tolist(),
                        dz_gt=g["dz"][ts].round(1).tolist(), stateA=round(A["state"], 2), stateB=round(B["state"], 2)))
json.dump(out, open("dis.json", "w"), indent=1)
for o in out: print(o)
