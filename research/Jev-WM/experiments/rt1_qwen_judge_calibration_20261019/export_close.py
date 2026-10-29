"""Export close-score SFT groups (plus contrasts) for human review. No judge calls."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "review_close_scores"
GROUPS = {
    "A_saturated_but_different": [22, 31, 21, 8, 0],
    "B_frozen_candidates_scored_high": [17, 4, 19],
    "C_near_identical_candidates": [1, 25, 10],
    "D_contrast_large_spread": [28, 13],
}
RUN = {i: ("dev_8b_s2" if i < 16 else "confirm_8b_s2") for g in GROUPS.values() for i in g}


def label(img, text, h=34):
    canvas = Image.new("RGB", (img.width, img.height + h), "white")
    canvas.paste(img, (0, h))
    ImageDraw.Draw(canvas).multiline_text((4, 2), text, fill="black", spacing=2)
    return canvas


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    idx = json.loads((ROOT / "cases/sft/index.json").read_text())
    scores = {}
    for run in set(RUN.values()):
        for r in json.loads((ROOT / f"results/{run}/scores.json").read_text()):
            scores[r["index"]] = r
    html = ["<html><meta charset='utf-8'><body style='font-family:sans-serif'>",
            "<h1>分数相近样本复核（Qwen3-VL-8B ×2）</h1>",
            "<p>每个视频：左上 GT（原始），其余为 4 条 SFT 候选。帧 0-1 为共享初始观测，2-11 为预测。"
            "标题：combined = 两顺序平均加权分；S/D/I = 结构/动态/交互 期望等级/4，括号内为 ref-first / pred-first。"
            "rep = 相邻帧完全重复次数（共 9 次转换）；lpips 仅为离线诊断。</p>"]
    summary = []
    for group, cases in GROUPS.items():
        html.append(f"<h2>{group}</h2>")
        for i in cases:
            with np.load(ROOT / f"cases/sft/{i:03d}.npz") as z:
                init, gt, gtc, preds = z["initial"], z["reference_raw"], z["reference_codec"], z["predictions"]
            r, meta = scores[i], idx[i]
            clips = [("GT raw", gt, "")]
            rows = []
            for k in range(4):
                s = r["scores"][f"sft{k}"]
                pq = s["per_question"]
                txt = (f"sft{k} combined {s['combined']:.3f}  rep {meta['frame_repeats'][k]}  lpips {meta['lpips'][k]:.3f}\n"
                       + "  ".join(f"{q[0].upper()} {np.mean(v):.2f}({v[0]:.2f}/{v[1]:.2f})" for q, v in pq.items()))
                clips.append((f"sft{k}", preds[k], txt))
                rows.append(dict(candidate=f"sft{k}", combined=s["combined"], per_order=s["per_order"],
                                 per_question=pq, frame_repeats=meta["frame_repeats"][k], lpips=meta["lpips"][k]))
            ctrl = {k: r["scores"][k]["combined"] for k in ["gt_raw", "gt_codec", "freeze_all", "freeze_half", "reverse", "black"]}
            gt_step = float(np.abs(np.diff(gt.astype(float), axis=0)).mean())
            cand_diff = [[float(np.abs(preds[a].astype(float) - preds[b].astype(float)).mean()) for b in range(4)] for a in range(4)]
            name = f"{group[0]}_{i:02d}"
            folder = OUT / name
            folder.mkdir()
            # Video: 2x3 grid, each frame labelled; initial frames shown in every tile.
            tiles_per_t = []
            for t in range(12):
                tiles = []
                for title, clip, txt in clips:
                    frame = init[t] if t < 2 else clip[t - 2]
                    head = f"{title}  t={t}{' (initial)' if t < 2 else ''}\n" + (txt.split(chr(10))[0] if txt else meta["task"][:60])
                    tiles.append(label(Image.fromarray(frame), head))
                w, h = tiles[0].size
                sheet = Image.new("RGB", (w * 3, h * 2), "white")
                for j, tile in enumerate(tiles):
                    sheet.paste(tile, ((j % 3) * w, (j // 3) * h))
                sheet.save(folder / f"grid_{t:02d}.png")
                tiles_per_t.append(sheet)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "2", "-i", str(folder / "grid_%02d.png"),
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            str(folder / "video.mp4")], check=True)
            for p in folder.glob("grid_*.png"):
                p.unlink()
            # Contact sheet: rows GT/sft0-3, columns future t=2..11 (half size).
            rows_img = []
            for title, clip, txt in clips:
                strip = Image.new("RGB", (160 * 10, 128), "white")
                for t in range(10):
                    strip.paste(Image.fromarray(clip[t]).resize((160, 128)), (160 * t, 0))
                rows_img.append(label(strip, (txt or f"GT raw   gt step |Δ| {gt_step:.1f}/255   task: {meta['task']}"), 36))
            sheet = Image.new("RGB", (rows_img[0].width, sum(x.height for x in rows_img)), "white")
            y = 0
            for x in rows_img:
                sheet.paste(x, (0, y))
                y += x.height
            sheet.save(folder / "contact_sheet.png")
            info = dict(index=i, id=meta["id"], task=meta["task"], group=group, judge_run=RUN[i],
                        sft=rows, controls=ctrl, gt_mean_step_abs_diff=gt_step,
                        candidate_pairwise_mean_abs_diff=cand_diff)
            (folder / "scores.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
            summary.append(info)
            html.append(f"<h3>{name}: {meta['task']}</h3><p>GT 平均帧间变化 {gt_step:.1f}/255；"
                        f"对照 combined: " + ", ".join(f"{k} {v:.3f}" for k, v in ctrl.items()) + "</p>")
            html.append("<table border=1 cellpadding=3><tr><th>cand</th><th>combined</th><th>ref-first</th><th>pred-first</th>"
                        "<th>S</th><th>D</th><th>I</th><th>rep</th><th>lpips</th></tr>")
            for row in rows:
                pq = row["per_question"]
                html.append(f"<tr><td>{row['candidate']}</td><td>{row['combined']:.3f}</td><td>{row['per_order'][0]:.3f}</td>"
                            f"<td>{row['per_order'][1]:.3f}</td>" + "".join(f"<td>{np.mean(pq[q]):.2f}</td>" for q in pq)
                            + f"<td>{row['frame_repeats']}</td><td>{row['lpips']:.3f}</td></tr>")
            html.append(f"</table><video src='{name}/video.mp4' controls loop width=960></video><br>"
                        f"<img src='{name}/contact_sheet.png' width=1200>")
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)


if __name__ == "__main__":
    main()
