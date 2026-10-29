"""Stage 2 (GPU): offline Qwen3-VL judges on within-group contrast pairs from stage 1.

Judges (reference = codec34k reconstruction of GT, not raw GT):
  state       per-frame QA "is the cube lifted / on table / missing", scored against h5 lift flag
  pair        Pref-GRPO style: REFERENCE + A + B, which is more faithful; both A/B orders
  point_recon V1 pointwise interaction prompt, unchanged, but with codec-recon reference
Results are appended to results/<judge>.jsonl; reruns skip finished keys.
"""
import sys, json, argparse, itertools, time
from pathlib import Path
import numpy as np, torch, h5py

SRC = "/media/tyshao/Jev-WM/experiments/robocasa_object_memspo_4gpu50_20261024/source/src"
sys.path.insert(0, SRC)
from futureworlds.qwen_judge import QwenJudge, INTRO, QUESTIONS, SUFFIX, to_image, ORDERS
from futureworlds.codec import CompressiveVQModelFSQ
from futureworlds.rewards import CodecWorker
from futureworlds.batch import TensorBatch
from futureworlds.data import read_case

OUT = Path(__file__).resolve().parent / "smoke"
E = Path("/media/tyshao/Jev-WM/experiments")
RUNS = {"v1": E / "robocasa_object_memspo_4gpu50_20261024", "v2": E / "robocasa_object_memspo_4gpu_v2_20261024"}
MANP = "/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"
MAN = {r["id"]: r for r in json.load(open(MANP))["cases"]}
CODEC = "/media/tyshao/fail_WM/FutureWorlds_SFT/previews_wm_codec34k_20261012/natural_bundle/robocasa/codec"
QWEN = "/media/tyshao/Jev-WM/paper_assets/qwen/Qwen3-VL-8B-Instruct"
X0, Y0, X1, Y1 = 130, 40, 310, 220
H = 32
TS = sorted(set(range(0, H, 2)) | {H - 1})
V1_SUFFIX = ('\nFocus on the RED CUBE relative to the gripper and tabletop. '
    'These are synchronized crops sampled every two frames, plus the final frame. '
    'If REFERENCE leaves the cube on the table, PREDICTION lifting or carrying it is a severe error, even if smooth. '
    'Likewise penalize failure to lift when REFERENCE actually lifts, unsupported attachment, wrong release timing, '
    'teleportation, disappearing or duplicated cubes. Do not infer grasp success from the task or arm motion alone. '
    'Answer one digit 0 to 4 only.')
STATE_Q = ("Two cropped frames from a robot simulation camera showing a gripper and a small RED CUBE. "
    "Image 1 is the start of the episode, with the cube resting on the table. Image 2 is a later frame. "
    "Question: in Image 2, is the red cube resting on the table, or lifted off the table (held up in the air)? "
    "Answer 0 if resting on the table, 1 if lifted off the table, 2 if no red cube is visible. "
    "Answer with a single digit and nothing else.")
PAIR_INTRO = ("You are comparing two robot video world-model predictions, A and B, against the REFERENCE, which is what "
    "the camera actually recorded. Both predictions start from the same two INITIAL frames and the same robot actions. "
    "Future frames are crops around the gripper and a small RED CUBE. The reference is evidence of what really "
    "happened, not an ideal success. Small blur or color differences are not errors.")
PAIR_Q = ("Question: which prediction reproduces the REFERENCE's red-cube state over time more faithfully: whether the "
    "cube stays on the table or is lifted and held, when it is lifted or released, and where it is? A prediction that "
    "lifts or carries the cube while the REFERENCE leaves it on the table is worse, and so is one that leaves it on the "
    "table while the REFERENCE lifts it. Answer 1 if A is more faithful, 2 if B is more faithful. "
    "Answer with a single digit and nothing else.")


def crop(x):
    return x[..., Y0:Y1, X0:X1, :]


def pairs_from_labels():
    lab = json.load(open(OUT / "labels.json"))
    out = []
    for g in lab["groups"]:
        for a, b in itertools.combinations(g["cands"], 2):
            if abs(a["agree"] - b["agree"]) >= 0.25:
                good, bad = (a, b) if a["agree"] > b["agree"] else (b, a)
                out.append(dict(run=g["run"], cursor=g["cursor"], id=g["id"], good=good["k"], bad=bad["k"],
                                gt_lift_frames=g["gt_lift_frames"], failure_type=g["failure_type"] or "success"))
    return out


def gt_lift(cid):
    with h5py.File(MAN[cid]["source_h5"]) as f:
        fl = f["events/flags"][:, 2]
    return fl[(np.arange(H) + 2) * 4].astype(int)


class Clips:
    def __init__(self):
        self.c = {}

    def __call__(self, run, cursor):
        key = (run, cursor)
        if key not in self.c:
            z = np.load(RUNS[run] / "judge_log" / f"{cursor:06d}_clips.npz")
            self.c = {key: dict(initial=z["initial"], reference=z["reference"], predictions=z["predictions"])}
        return self.c[key]


@torch.no_grad()
def reconstruct(codec, worker, cid, device):
    images, _, _ = read_case(MANP, MAN[cid], 52, H)
    fr = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to(device).float() / 255
    first = fr[:1]
    ctx, initial = codec.tokenize(first.repeat(2, 1, 1, 1)[None].contiguous())
    parts = [initial]
    for s in range(1, len(fr), 8):
        parts.append(codec.tokenize(torch.cat([first, fr[s:s + 8]])[None].contiguous())[1])
    dyn = torch.cat(parts, 1)[0]
    pix = worker.detokenize(TensorBatch({"tokens": dyn[2:][None], "ctx_tokens": ctx}), None).batch["pixels"][0, 1:]
    rec = np.rint(pix.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    assert rec.shape == (H, 256, 320, 3), rec.shape
    return rec, images[2:2 + H]


def done_keys(path):
    if not path.exists():
        return set()
    return {json.loads(l)["key"] for l in path.read_text().splitlines() if l.strip()}


def digit_probs(judge, items, digits):
    lg = judge.level_logits(items)[:, digits]
    p = np.exp(lg - lg.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="state,pair,point_recon")
    ap.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    P = pairs_from_labels()
    if a.limit:
        P = P[:: max(1, len(P) // a.limit)][: a.limit]
    groups = sorted({(p["run"], p["cursor"], p["id"]) for p in P})
    cands = sorted({(p["run"], p["cursor"], k) for p in P for k in (p["good"], p["bad"])})
    print(f"pairs {len(P)} groups {len(groups)} candidates {len(cands)} episodes {len({g[2] for g in groups})}", flush=True)
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "pairs.json").write_text(json.dumps(P))

    codec = CompressiveVQModelFSQ.from_pretrained(CODEC, local_files_only=True).to(a.device).eval()
    worker = CodecWorker(codec, None)
    recon, lift = {}, {}
    rec_err = []
    for cid in sorted({g[2] for g in groups}):
        recon[cid], raw = reconstruct(codec, worker, cid, a.device)
        lift[cid] = gt_lift(cid)
        rec_err.append(np.abs(recon[cid].astype(float) - raw).mean())
    del codec, worker
    torch.cuda.empty_cache()
    print(f"codec recon mean abs error {np.mean(rec_err):.2f}/255 over {len(recon)} episodes", flush=True)
    clips = Clips()
    for run, cursor, cid in groups[:3]:
        assert np.array_equal(clips(run, cursor)["reference"], read_case(MANP, MAN[cid], 52, H)[0][2:2 + H]), "clip/case mismatch"

    judge = QwenJudge(QWEN, a.device, scale=2, batch_size=1)
    img = lambda f: to_image(np.ascontiguousarray(f), judge.scale)

    def run_judge(name, keyed_items, digits, bs):
        path = OUT / "results" / f"{name}.jsonl"
        done = done_keys(path)
        todo = [(k, m) for k, m in keyed_items if k not in done]
        print(f"[{name}] {len(todo)} calls to run ({len(done)} done)", flush=True)
        judge.batch_size = bs
        t0 = time.time()
        with path.open("a") as fh:
            for i in range(0, len(todo), bs):
                chunk = todo[i:i + bs]
                probs = digit_probs(judge, [m() for _, m in chunk], digits)
                for (k, _), p in zip(chunk, probs):
                    fh.write(json.dumps(dict(key=k, p=p.tolist())) + "\n")
                fh.flush()
                n = i + len(chunk)
                if n % (bs * 20) < bs or n == len(todo):
                    print(f"[{name}] {n}/{len(todo)} {(time.time() - t0) / n:.2f}s/call", flush=True)

    js = a.judges.split(",")
    if "state" in js:
        items = []
        for run, cursor, cid in groups:
            c = clips(run, cursor)
            init = crop(c["initial"][-1])
            for src, frames in (("raw", c["reference"]), ("recon", recon[cid])):
                if run == "v2" and ("v1", cursor, cid) in groups:
                    continue  # identical GT frames already queued from v1
                for t in TS:
                    items.append((f"gt|{src}|{cid}|{t}", lambda i=init, f=crop(frames[t]): [dict(role="user", content=[
                        dict(type="image", image=img(i)), dict(type="image", image=img(f)), dict(type="text", text=STATE_Q)])]))
        for run, cursor, k in cands:
            c = clips(run, cursor)
            init = crop(c["initial"][-1])
            for t in TS:
                items.append((f"pred|{run}|{cursor}|{k}|{t}", lambda i=init, f=crop(c["predictions"][k, t]): [dict(role="user", content=[
                    dict(type="image", image=img(i)), dict(type="image", image=img(f)), dict(type="text", text=STATE_Q)])]))
        items = list({k: m for k, m in items}.items())
        run_judge("state", items, [0, 1, 2], 8)

    if "pair" in js:
        items = []
        for p in P:
            c = clips(p["run"], p["cursor"])
            ref = crop(recon[p["id"]][TS])
            for order in ("good_is_A", "good_is_B"):
                A, B = (p["good"], p["bad"]) if order == "good_is_A" else (p["bad"], p["good"])
                def build(c=c, ref=ref, A=A, B=B):
                    content = [dict(type="text", text=PAIR_INTRO)]
                    for t, f in enumerate(c["initial"]):
                        content += [dict(type="text", text=f"INITIAL frame {t}:"), dict(type="image", image=img(f))]
                    content.append(dict(type="text", text=f"Future timestamps 1 to {len(TS)} follow; each shows REFERENCE, A, B."))
                    pa, pb = crop(c["predictions"][A][TS]), crop(c["predictions"][B][TS])
                    for t in range(len(TS)):
                        content += [dict(type="text", text=f"t={t + 1} REFERENCE:"), dict(type="image", image=img(ref[t])),
                                    dict(type="text", text=f"t={t + 1} A:"), dict(type="image", image=img(pa[t])),
                                    dict(type="text", text=f"t={t + 1} B:"), dict(type="image", image=img(pb[t]))]
                    content.append(dict(type="text", text=PAIR_Q))
                    return [dict(role="user", content=content)]
                items.append((f"{p['run']}|{p['cursor']}|{p['good']}|{p['bad']}|{order}", build))
        run_judge("pair", items, [1, 2], 1)

    if "point_recon" in js:
        items = []
        for run, cursor, k in cands:
            c = clips(run, cursor)
            cid = next(g[2] for g in groups if g[:2] == (run, cursor))
            for order in ORDERS:
                def build(c=c, cid=cid, k=k, order=order):
                    m = judge.messages(c["initial"], crop(recon[cid][TS]), crop(c["predictions"][k][TS]),
                                       "Pick up the cube and place it at the goal.", order, "interaction")
                    m[0]["content"][-1]["text"] += V1_SUFFIX
                    return m
                items.append((f"{run}|{cursor}|{k}|{order}", build))
        run_judge("point_recon", items, [0, 1, 2, 3, 4], 1)
    (OUT / "lift_gt.json").write_text(json.dumps({k: v.tolist() for k, v in lift.items()}))
    print("STAGE2 DONE", flush=True)


if __name__ == "__main__":
    main()
