"""Stage A (GPU): per-frame Qwen state QA on GT frames, scored against simulator flags.

Frames are the real 256x320 reference frames of train_natural windows (reference frame k
= h5 index (2+k)*4, same alignment as training). Category per frame from h5 events/flags:
  lifted          lift flag and cube >= 3 cm above its start height
  grasp_nolift    grasp flag, no lift flag, cube < 1 cm above start
  contact_nolift  contact flag only, cube < 1 cm above start
  free_closed     no contact, gripper commanded closed (actions/raw[:,6] = +1; grasp_miss look-alike)
  free_open       no contact, gripper commanded open
Question variants (p_lift = probability of the "lifted" digit, renormalized over the answer digits):
  v0     current training question: 0 table / 1 lifted / 2 missing, standard crop, scale 2
  v1     4-way: 0 untouched on table / 1 on table but gripper touching or closed on/next to it /
         2 lifted clear of the table / 3 missing; standard crop, scale 2
  v2     v1 question on a tight crop around the cube (start frame: start cube position; later
         frame: tracked cube position in that frame, start position if not found), scale 3
  v3     4-way, with the cue that a cube held between the fingers AND moved away from its start
         position is lifted; standard crop, scale 2
  v4     v3 plus a yellow box drawn on Image 2 at the cube's start position (tracked in the last
         context frame, identical for every candidate); standard crop, scale 2
         (v3/v4 answer "lifted" for almost every frame: the long "Signs:" clause biases option 2)
  v5     v1 options unchanged, plus one closing sentence: carried away from the start = lifted
  v6     v1 question plus the v4 start box and a neutral one-sentence note
Results append to results/stage_a.jsonl; reruns skip finished keys.
"""
import argparse, collections, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from PIL import ImageDraw

SRC = "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/source/src"
sys.path.insert(0, SRC)
from futureworlds.data import read_case
from futureworlds.qwen_judge import QwenJudge, to_image
from futureworlds.robocasa_object_reward import track
from futureworlds.robocasa_state_reward import STATE_Q

OUT = Path(__file__).resolve().parent
MANP = "/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"
QWEN = "/media/tyshao/Jev-WM/paper_assets/qwen/Qwen3-VL-8B-Instruct"
H = 32
IDX = (np.arange(H) + 2) * 4
X0, Y0, X1, Y1 = 130, 40, 310, 220
TIGHT = 64  # half-size of the cube-centered crop (128x128 px)

STATE_Q4 = ("Two cropped frames from a robot simulation camera showing a gripper and a small RED CUBE. "
    "Image 1 is the start of the episode, with the cube resting on the table. Image 2 is a later frame. "
    "Question: in Image 2, what is the state of the red cube?\n"
    "0 = resting on the table, the gripper is not touching it.\n"
    "1 = still resting on the table, but the gripper is touching it, closed around it, or closed right next to it. "
    "The bottom of the cube is still on the table surface.\n"
    "2 = lifted: the cube is held up in the air, clearly above the table surface and higher than in Image 1.\n"
    "3 = no red cube is visible.\n"
    "A closed gripper near the cube does not mean the cube is lifted. "
    "Answer 2 only if the cube itself is visibly off the table. "
    "Answer with a single digit and nothing else.")

STATE_Q4_MOVED = ("Two cropped frames from a robot simulation camera showing a gripper and a small RED CUBE. "
    "Image 1 is the start of the episode, with the cube resting on the table. Image 2 is a later frame.{box} "
    "Question: in Image 2, what is the state of the red cube?\n"
    "0 = resting on the table at its start position, the gripper is not touching it.\n"
    "1 = still on the table: the gripper may be touching it, closed around it, or closed right next to it, "
    "but the cube has not been raised and is still at its start position.\n"
    "2 = lifted: the gripper is holding the cube up above the table. Signs: the cube is between the gripper "
    "fingers AND has moved away from its start position, or the cube is visibly higher than the table surface.\n"
    "3 = no red cube is visible.\n"
    "A closed gripper near the cube does not by itself mean the cube is lifted. "
    "Answer with a single digit and nothing else.")
BOX_NOTE = " The yellow box in Image 2 marks where the cube was in Image 1."
LAST = "Answer with a single digit and nothing else."
assert STATE_Q4.endswith(LAST)
STATE_Q4_CARRIED = STATE_Q4[:-len(LAST)] + (
    "If the gripper has carried the cube away from where it was in Image 1, the cube is lifted (2). " + LAST)
STATE_Q4_BOX = STATE_Q4.replace("Image 2 is a later frame.", "Image 2 is a later frame." + BOX_NOTE)
assert STATE_Q4_BOX != STATE_Q4


def category(flags, dz, grip):
    c, g, l = flags[:3]
    if l:
        return "lifted" if dz >= 3 else None
    if dz >= 1:
        return None
    if g:
        return "grasp_nolift"
    if c:
        return "contact_nolift"
    return "free_closed" if grip > 0 else "free_open"


def sample_frames(per_cat, max_per_episode, seed):
    rows = json.load(open(MANP))["cases"]
    rng = np.random.default_rng(seed)
    pool = collections.defaultdict(list)
    for r in rows:
        with h5py.File(r["source_h5"]) as f:
            fl = f["events/flags"][:][IDX]
            z = f["observations/object_pose_wxyz"][:, 2]
            grip = f["actions/raw"][:, 6]  # gripper command: -1 open, +1 close
        dz = (z[IDX] - z[0]) * 100
        per = collections.defaultdict(list)
        for t in range(H):
            cat = category(fl[t], dz[t], grip[min(IDX[t], len(grip) - 1)])
            if cat:
                per[cat].append(t)
        for cat, ts in per.items():
            for t in rng.permutation(ts)[:max_per_episode]:
                pool[cat].append(dict(id=r["id"], t=int(t), cat=cat, type=r["failure_type"] or "success",
                                      dz_cm=float(dz[t])))
    out = []
    for cat, items in sorted(pool.items()):
        pick = rng.permutation(len(items))[:per_cat]
        out += [items[i] for i in pick]
        print(f"  {cat:15s} pool {len(items):5d} -> {len(pick)}", flush=True)
    return out, {r["id"]: r for r in rows}


def tight(frame, xy):
    x, y = int(round(xy[0])), int(round(xy[1]))
    h, w = frame.shape[:2]
    x = min(max(x, TIGHT), w - TIGHT); y = min(max(y, TIGHT), h - TIGHT)
    return np.ascontiguousarray(frame[y - TIGHT:y + TIGHT, x - TIGHT:x + TIGHT])


def with_start_box(image, start, scale=2, half=9):
    """Yellow outline at the cube's start position, in crop coordinates of the scaled image."""
    xy = track(start[None])[0]["xy"]
    if xy is None:
        return image
    x, y = (xy[0] - X0) * scale, (xy[1] - Y0) * scale
    image = image.copy()
    ImageDraw.Draw(image).rectangle([x - half * scale, y - half * scale, x + half * scale, y + half * scale],
                                    outline=(255, 230, 0), width=2)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cat", type=int, default=120)
    ap.add_argument("--variants", default="v0,v1,v3,v4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    (OUT / "results").mkdir(exist_ok=True)
    spath = OUT / "stage_a_frames.json"
    if spath.exists():
        frames = json.loads(spath.read_text()); index = {r["id"]: r for r in json.load(open(MANP))["cases"]}
    else:
        frames, index = sample_frames(a.per_cat, 2, 20261027)
        spath.write_text(json.dumps(frames, indent=1))
    print(f"{len(frames)} frames from {len({f['id'] for f in frames})} episodes", flush=True)

    res = OUT / "results" / "stage_a.jsonl"
    done = {json.loads(l)["key"] for l in res.read_text().splitlines() if l.strip()} if res.exists() else set()
    judge = QwenJudge(QWEN, a.device, scale=2, batch_size=a.batch_size)
    clips = {}

    def clip(cid):
        if cid not in clips:
            images, _, _ = read_case(MANP, index[cid], 52, H)
            clips.clear()
            clips[cid] = images
        return clips[cid]

    todo = []
    for f in sorted(frames, key=lambda f: f["id"]):
        for v in a.variants.split(","):
            key = f"{v}|{f['id']}|{f['t']}"
            if key not in done:
                todo.append((key, v, f))
    print(f"{len(todo)} calls to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with res.open("a") as fh:
        for i in range(0, len(todo), a.batch_size):
            chunk = todo[i:i + a.batch_size]
            items, kinds = [], []
            for key, v, f in chunk:
                images = clip(f["id"])
                start, frame = images[1], images[2 + f["t"]]
                if v == "v2":
                    xy0 = track(start[None])[0]["xy"]
                    xy = track(frame[None])[0]["xy"] or xy0
                    s, p = to_image(tight(start, xy0), 3), to_image(tight(frame, xy), 3)
                else:
                    s = to_image(np.ascontiguousarray(start[Y0:Y1, X0:X1]), 2)
                    p = to_image(np.ascontiguousarray(frame[Y0:Y1, X0:X1]), 2)
                if v in ("v4", "v6"):
                    p = with_start_box(p, start)
                q = {"v0": STATE_Q, "v1": STATE_Q4, "v2": STATE_Q4,
                     "v3": STATE_Q4_MOVED.format(box=""), "v4": STATE_Q4_MOVED.format(box=BOX_NOTE),
                     "v5": STATE_Q4_CARRIED, "v6": STATE_Q4_BOX}[v]
                items.append([dict(role="user", content=[dict(type="image", image=s), dict(type="image", image=p),
                                                         dict(type="text", text=q)])])
                kinds.append(3 if v == "v0" else 4)
            logits = judge.level_logits(items)
            for (key, v, f), lg, n in zip(chunk, logits, kinds):
                x = lg[:n]; pr = np.exp(x - x.max()); pr /= pr.sum()
                fh.write(json.dumps(dict(key=key, variant=v, id=f["id"], t=f["t"], cat=f["cat"], type=f["type"],
                                         p=pr.tolist(), p_lift=float(pr[1] if v == "v0" else pr[2]))) + "\n")
            fh.flush()
            n = i + len(chunk)
            if n % (a.batch_size * 25) < a.batch_size or n == len(todo):
                print(f"{n}/{len(todo)} {(time.time() - t0) / n:.2f}s/call", flush=True)
    print("STAGE A DONE", flush=True)


if __name__ == "__main__":
    main()
