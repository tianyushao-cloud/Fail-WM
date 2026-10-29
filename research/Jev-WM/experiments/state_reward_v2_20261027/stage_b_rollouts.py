"""Stage B (GPU): state QA variants on real RL rollouts (8-GPU run judge_log clips, 8 candidates/group).

v0 is not recomputed: its per-frame p_lift is in the training judge log (same question, crop, scale).
Groups: every GT-no-lift window whose candidates mix tracker-rise and no-rise (the fake-lift contrast
the reward must rank), plus every third GT-lift window as a check that true lifts are not punished.
Tracker lift per predicted frame: cube rises > 8 px above its start height (stage-1 calibration).
Results append to results/stage_b.jsonl, one line per (variant, cursor, candidate); reruns skip done keys.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import stage_a_gt_frames as A
from futureworlds.qwen_judge import QwenJudge, to_image

RUN = Path("/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log")
THR = 8
TS = sorted(set(range(0, 32, 2)) | {31})
QUESTIONS = {"v1": (A.STATE_Q4, False), "v5": (A.STATE_Q4_CARRIED, False), "v6": (A.STATE_Q4_BOX, True)}


def tracker_lift(d):
    """[n, 32] per-frame tracker lift and the GT start height, from the logged tracks."""
    out = []
    for t in d["tracks"]:
        y0 = next(g["xy"][1] for g in t["gt"] if g["xy"] is not None)
        ys = np.array([np.nan if p["xy"] is None else p["xy"][1] for p in t["pred"]])
        out.append(np.where(np.isnan(ys), 0, (y0 - ys) > THR).astype(int))
    return np.array(out)


def select():
    groups = []
    for f in sorted(RUN.glob("[0-9]*[0-9].json")):
        d = json.loads(f.read_text())
        rise = tracker_lift(d).sum(1) >= 3
        if not any(d["gt_lift"]):
            if rise.any() and not rise.all():
                groups.append(d["cursor"])
        elif d["cursor"] % 3 == 0:
            groups.append(d["cursor"])
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="v1,v6")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    gpath = D / "stage_b_groups.json"
    if not gpath.exists():
        gpath.write_text(json.dumps(select()))
    groups = json.loads(gpath.read_text())
    res = D / "results" / "stage_b.jsonl"
    done = {json.loads(l)["key"] for l in res.read_text().splitlines() if l.strip()} if res.exists() else set()
    todo = [(v, c) for c in groups for v in a.variants.split(",") if not all(f"{v}|{c}|{k}" in done for k in range(8))]
    print(f"{len(groups)} groups; {len(todo)} (variant, group) jobs to run", flush=True)
    judge = QwenJudge(A.QWEN, a.device, scale=2, batch_size=a.batch_size)
    t0 = time.time()
    with res.open("a") as fh:
        for j, (v, c) in enumerate(todo):
            q, box = QUESTIONS[v]
            z = np.load(RUN / f"{c:06d}_clips.npz")
            start_full, preds = z["initial"][-1], z["predictions"]
            s = to_image(np.ascontiguousarray(start_full[A.Y0:A.Y1, A.X0:A.X1]), 2)
            items = []
            for k in range(len(preds)):
                for t in TS:
                    p = to_image(np.ascontiguousarray(preds[k, t, A.Y0:A.Y1, A.X0:A.X1]), 2)
                    if box:
                        p = A.with_start_box(p, start_full)
                    items.append([dict(role="user", content=[dict(type="image", image=s), dict(type="image", image=p),
                                                             dict(type="text", text=q)])])
            lg = judge.level_logits(items)[:, :4].reshape(len(preds), len(TS), 4)
            pr = np.exp(lg - lg.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
            for k in range(len(preds)):
                fh.write(json.dumps(dict(key=f"{v}|{c}|{k}", variant=v, cursor=c, k=k,
                                         p=pr[k].round(5).tolist(), p_lift=pr[k, :, 2].round(5).tolist())) + "\n")
            fh.flush()
            if (j + 1) % 10 == 0 or j + 1 == len(todo):
                el = time.time() - t0
                print(f"{j + 1}/{len(todo)} {el / (j + 1):.1f}s/group, eta {el / (j + 1) * (len(todo) - j - 1) / 60:.0f} min", flush=True)
    print("STAGE B DONE", flush=True)


if __name__ == "__main__":
    main()
