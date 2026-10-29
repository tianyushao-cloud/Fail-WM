"""Render GT | SFT | RL side-by-side review clips (full frame + cube zoom) with per-frame lift badges.

Usage: render_review.py OUT_DIR LABEL_A LABEL_B case_id:title [case_id:title ...]   (val split)
Badge per column: simulator lift flag (GT) or tracker lift (rise > 8 px) for predictions.
Also writes OUT_DIR/index.html and a 5-frame zoom contact sheet per case.
"""
import html, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

R = Path("/media/tyshao/Jev-WM/experiments/sibling_action_eval_20261025")
sys.path.insert(0, "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/source/src")
from futureworlds.data import read_case

MANP = "/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/val.json"
MAN = {r["id"]: r for r in json.load(open(MANP))["cases"]}
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
A, B = sys.argv[2], sys.argv[3]
CASES = [c.split(":", 1) for c in sys.argv[4:]]
X0, Y0, X1, Y1 = 130, 40, 310, 220
GAP = 8
CT = {r["id"]: r for r in json.load(open(R / "case_table.json"))}


def zoom(f):
    return np.asarray(Image.fromarray(np.ascontiguousarray(f[Y0:Y1, X0:X1])).resize((320, 320), Image.BICUBIC))


def lift_track(label, cid):
    d = json.loads((R / "results" / label / f"{cid}.json").read_text())
    r = np.array([np.nan if v is None else v for v in d["pred_rise_px"]], float)
    return np.where(np.isnan(r), 0, r > 8).astype(int), np.array(d["gt_lift"])


def badge(img, text, on):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 319, 22], fill=(0, 0, 0))
    dr.text((6, 5), text, fill=(255, 255, 255))
    if on is not None:
        dr.rectangle([236, 3, 316, 19], fill=(220, 40, 40) if on else (40, 140, 60))
        dr.text((242, 5), "LIFTED" if on else "on table", fill=(255, 255, 255))
    return np.asarray(im)


def column(frame, title, on):
    top = badge(frame.copy(), title, on)
    return np.concatenate([top, np.full((GAP, 320, 3), 255, np.uint8), zoom(frame)])


rows_html = []
for cid, title in CASES:
    images, _, _ = read_case(MANP, MAN[cid], 52, 32)
    pa = np.load(R / "results" / A / f"{cid}_pred.npz")["prediction"]
    pb = np.load(R / "results" / B / f"{cid}_pred.npz")["prediction"]
    la, gt = lift_track(A, cid); lb, _ = lift_track(B, cid)
    gt_all = np.concatenate([[0, 0], gt]); la = np.concatenate([[0, 0], la]); lb = np.concatenate([[0, 0], lb])
    frames = []
    for t in range(34):
        ctx = t < 2
        cols = [column(images[t], f"GT  t={t}" + (" (context)" if ctx else ""), None if ctx else gt_all[t]),
                column(images[t] if ctx else pa[t - 2], "SFT", None if ctx else la[t]),
                column(images[t] if ctx else pb[t - 2], "RL v0 step30", None if ctx else lb[t])]
        sep = np.full((cols[0].shape[0], GAP, 3), 255, np.uint8)
        frames.append(np.concatenate([cols[0], sep, cols[1], sep, cols[2]], 1))
    h, w = frames[0].shape[:2]
    mp4 = OUT / f"{cid}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "4",
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "slow",
                    "-movflags", "+faststart", str(mp4)], input=np.stack(frames).tobytes(), check=True)
    picks = [2, 10, 18, 26, 33]
    strip = lambda seq: np.concatenate([np.concatenate([zoom(seq[t]), np.full((320, 4, 3), 255, np.uint8)], 1) for t in picks], 1)
    full_a = np.concatenate([images[:2], pa]); full_b = np.concatenate([images[:2], pb])
    sheet = np.concatenate([strip(images), np.zeros((6, strip(images).shape[1], 3), np.uint8), strip(full_a),
                            np.zeros((6, strip(images).shape[1], 3), np.uint8), strip(full_b)])
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(OUT / f"{cid}_sheet.jpg", quality=85)
    ct = CT[cid]
    m = {l: json.loads((R / "results" / l / "frame_metrics_roi.json").read_text())[cid] for l in (A, B)}
    lp = {l: np.mean(m[l]["lpips"]) for l in m}
    rows_html.append(f"""<section><h2>{html.escape(title)}</h2>
<p><b>{cid}</b> · 仿真结果 <b>{ct['type']}</b> · 夹爪指令 <code>{ct['cmd_seq']}</code>（o 开 / h 半闭 / F 全闭，每 4 帧一个）· 末端开度 {ct['final_width_cm']:.1f} cm<br>
窗口内 GT 抬起：<b>{'是' if gt.any() else '否'}</b> · SFT 预测抬起：<b>{'是' if la.sum() >= 3 else '否'}</b> · RL 预测抬起：<b>{'是' if lb.sum() >= 3 else '否'}</b>
· ROI LPIPS(32f) SFT {lp[A]:.3f} / RL {lp[B]:.3f}</p>
<video src="{cid}.mp4" controls loop muted playsinline width="984"></video>
<p>关键帧放大（行：GT / SFT / RL；列：t=2,10,18,26,33）</p><img src="{cid}_sheet.jpg" width="984"></section>""")
    print(cid, mp4.stat().st_size // 1024, "KB", flush=True)

(OUT / "index.html").write_text("""<!doctype html><meta charset="utf-8"><title>v0 step30 典型样例</title>
<style>body{font-family:sans-serif;max-width:1020px;margin:20px auto;color:#222}section{margin:34px 0;border-top:1px solid #ccc}code{font-size:12px}</style>
<h1>RoboCasa WM：SFT vs RL（v0 state reward, step30），val sibling 测试</h1>
<p>每段视频三列：GT（仿真器真值）| SFT | RL step30。上排整帧，下排红块区域放大。前 2 帧是真实上下文帧，之后 32 帧是模型预测（beam 4，确定性解码）。
右上角标签：GT 列为仿真器 lift 标记；预测列为 tracker 判定（红块比起始位置高 8 px 以上）。播放 4 fps。</p>
""" + "\n".join(rows_html))
print("index written")
