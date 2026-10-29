"""Render GT | SFT clips (full frame + zoomed cube crop) and contact sheets for chosen cases."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parent
sys.path.insert(0, "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_3gpu_20261025/source/src")
from futureworlds.data import read_case

MANP = "/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/val.json"
MAN = {r["id"]: r for r in json.load(open(MANP))["cases"]}
CASES = sys.argv[1].split(",")
OUT = R / "report_media"
OUT.mkdir(exist_ok=True)
X0, Y0, X1, Y1 = 130, 40, 310, 220
GAP = 8


def zoom(f):
    return np.asarray(Image.fromarray(f[Y0:Y1, X0:X1]).resize((320, 320), Image.BICUBIC))


def compose(gt, pr):
    w = 320 * 2 + GAP
    top = np.full((256, w, 3), 255, np.uint8)
    top[:, :320] = gt
    top[:, 320 + GAP:] = pr
    bot = np.full((320, w, 3), 255, np.uint8)
    bot[:, :320] = zoom(gt)
    bot[:, 320 + GAP:] = zoom(pr)
    return np.concatenate([top, np.full((GAP, w, 3), 255, np.uint8), bot])


for cid in CASES:
    images, _, _ = read_case(MANP, MAN[cid], 52, 32)
    pred = np.load(R / "results/sft" / f"{cid}_pred.npz")["prediction"]
    gt = images[:34]
    pr = np.concatenate([images[:2], pred])  # both columns start from the same two real frames
    frames = [compose(g, p) for g, p in zip(gt, pr)]
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "4",
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow",
           "-movflags", "+faststart", str(OUT / f"{cid}.mp4")]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)
    Image.fromarray(frames[-1]).save(OUT / f"{cid}_poster.jpg", quality=85)
    # contact sheet for review: frames 2,10,18,26,33 (zoom rows only)
    picks = [2, 10, 18, 26, 33]
    sheet = np.concatenate([np.concatenate([zoom(gt[t]), np.full((320, 4, 3), 255, np.uint8)], 1) for t in picks], 1)
    sheet2 = np.concatenate([np.concatenate([zoom(pr[t]), np.full((320, 4, 3), 255, np.uint8)], 1) for t in picks], 1)
    Image.fromarray(np.concatenate([sheet, np.full((6, sheet.shape[1], 3), 0, np.uint8), sheet2])).resize(
        (sheet.shape[1] // 2, (sheet.shape[0] * 2 + 6) // 2)).save(OUT / f"{cid}_sheet.jpg", quality=85)
    print(cid, (OUT / f"{cid}.mp4").stat().st_size // 1024, "KB", flush=True)
