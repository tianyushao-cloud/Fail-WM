"""Export test-split SFT vs RL videos for windows with the largest improvement, per RL arm.

Selection uses only offline metrics (32-frame LPIPS gain, token-static reduction) against SFT on
identical windows; the RL reward never saw these metrics. Each video shows GT | SFT | FM | A5.
"""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

E = Path("/media/tyshao/Jev-WM/experiments")
SPLIT = "test"
RUNS = {
    "SFT": E / f"rt1_qwen_a5_h32_20261020/evaluation/{SPLIT}/sft",
    "FM": E / f"rt1_qwen_fm_h32_20261020/evaluation/{SPLIT}/rl",
    "A5": E / f"rt1_qwen_a5_h32_20261020/evaluation/{SPLIT}/rl",
}
NAMES = {"FM": "Qwen RL (fidelity+motion)", "A5": "Qwen RL (4 axes)", "SFT": "SFT", "GT": "GT"}
OUT = E / "rt1_qwen_a5_h32_20261020/review_best_test"
PER_CRITERION = 3


def load(run, i):
    with np.load(RUNS[run] / f"case_{i:06d}" / "prediction.npz") as z:
        pred = z["prediction"]
        pred = (np.clip(pred, 0, 1) * 255).round().astype(np.uint8) if pred.dtype != np.uint8 else pred
        return pred, z["tokens"][0], z["truth"], z["observations"]


def static(tokens):
    same = np.array([np.array_equal(a, b) for a, b in zip(tokens[1:], tokens[:-1])])
    return same


def tile(frame, title, sub, w=320, h=256, head=36):
    img = Image.new("RGB", (w, h + head), "white")
    img.paste(Image.fromarray(frame), (0, head))
    d = ImageDraw.Draw(img)
    d.text((4, 2), title, fill="black")
    d.text((4, 18), sub, fill=(200, 0, 0) if "FROZEN" in sub else "black")
    return img


def main():
    summ = {k: json.loads((v / "summary.json").read_text()) for k, v in RUNS.items()}
    ids = [x["id"] for x in summ["SFT"]["per_case"]]
    for s in summ.values():
        assert [x["id"] for x in s["per_case"]] == ids
    n = len(ids)
    manifest = json.loads((E / f"rt1_memspo200_upstream_20261017/data/{SPLIT}/manifest.json").read_text())
    instruction = {c["id"]: c["instruction"] for c in manifest["cases"]}
    lp = {k: np.array([x["metrics"]["32"]["lpips"] for x in s["per_case"]]) for k, s in summ.items()}
    st = {k: np.array([static(load(k, i)[1]).mean() for i in range(n)]) for k in RUNS}
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    picks = []
    for arm in ["FM", "A5"]:
        gain = lp["SFT"] - lp[arm]
        drop = st["SFT"] - st[arm]
        chosen = []
        for crit, score in [("LPIPS提升", gain), ("停滞减少", drop)]:
            order = [i for i in np.argsort(-score, kind="stable") if i not in chosen]
            for i in order[:PER_CRITERION]:
                chosen.append(int(i))
                picks.append((arm, crit, int(i)))
    rows = []
    for arm, crit, i in picks:
        preds, toks = {}, {}
        for k in RUNS:
            preds[k], toks[k], truth, obs = load(k, i)
        task = instruction[ids[i]]
        name = f"{arm}_{'lpips' if crit == 'LPIPS提升' else 'stall'}_case{i:02d}"
        folder = OUT / name
        folder.mkdir()
        order = ["GT", "SFT", "FM", "A5"]
        sts = {k: static(toks[k]) for k in RUNS}
        for t in range(34):
            tiles = []
            for k in order:
                if t < 2:
                    frame, sub = obs[t], f"t={t} initial observation"
                else:
                    f = t - 2
                    frame = truth[f] if k == "GT" else preds[k][f]
                    frozen = k != "GT" and f > 0 and sts[k][f - 1]
                    lps = "" if k == "GT" else f" LPIPS32 {lp[k][i]:.3f} static {st[k][i]:.0%}"
                    sub = f"t={t}{'  FROZEN' if frozen else ''}{lps}"
                tiles.append(tile(frame, NAMES[k] + ("  <== selected arm" if k == arm else ""), sub))
            sheet = Image.new("RGB", (tiles[0].width * 4, tiles[0].height), "white")
            for j, tl in enumerate(tiles):
                sheet.paste(tl, (j * tiles[0].width, 0))
            sheet.save(folder / f"f_{t:02d}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "4", "-i", str(folder / "f_%02d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(folder / "video.mp4")], check=True)
        for p in folder.glob("f_*.png"):
            p.unlink()
        # Contact sheet: every 4th future frame, rows GT/SFT/FM/A5.
        ts = list(range(3, 32, 4))
        sheet = Image.new("RGB", (160 * len(ts), 128 * 4 + 20 * 4), "white")
        d = ImageDraw.Draw(sheet)
        for r, k in enumerate(order):
            y = r * 148
            d.text((4, y + 2), NAMES[k] + ("" if k == "GT" else f"   LPIPS32 {lp[k][i]:.3f}   static {st[k][i]:.0%}"), fill="black")
            for j, f in enumerate(ts):
                frame = truth[f] if k == "GT" else preds[k][f]
                sheet.paste(Image.fromarray(frame).resize((160, 128)), (160 * j, y + 20))
        sheet.save(folder / "contact_sheet.png")
        row = dict(folder=name, arm=arm, criterion=crit, case=i, id=ids[i], task=task,
                   lpips32={k: float(lp[k][i]) for k in RUNS}, static={k: float(st[k][i]) for k in RUNS})
        (folder / "info.json").write_text(json.dumps(row, indent=1, ensure_ascii=False) + "\n")
        rows.append(row)
    html = ["<html><meta charset='utf-8'><body style='font-family:sans-serif'>",
            "<h1>测试集：两条 Qwen RL 线中提升最大的样本（step100）</h1>",
            "<p>每个视频 4 列：GT | SFT | Qwen RL 二维（保真+运动持续）| Qwen RL 四维；4fps，t=0-1 为共享初始观测，t=2-33 为自主预测的 32 帧。"
            "红字 FROZEN = 该帧视觉 token 与上一帧完全相同。按 32 帧 LPIPS 提升、静止帧比例下降各取前 3（相对 SFT），只用离线指标挑选。</p>"]
    for arm in ["FM", "A5"]:
        html.append(f"<h2>{NAMES[arm]}</h2>")
        for r in [x for x in rows if x["arm"] == arm]:
            html.append(f"<h3>{r['folder']}：{r['criterion']}，{r['task']}</h3><p>"
                        + "；".join(f"{NAMES[k]} LPIPS32 {r['lpips32'][k]:.3f}, 静止 {r['static'][k]:.0%}" for k in RUNS) + "</p>"
                        f"<video src='{r['folder']}/video.mp4' controls loop width=1280></video><br>"
                        f"<img src='{r['folder']}/contact_sheet.png' width=1280>")
    (OUT / "index.html").write_text("\n".join(html) + "</body></html>")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)
    for r in rows:
        print(r["folder"], r["task"][:40], {k: round(v, 3) for k, v in r["lpips32"].items()}, {k: round(v, 2) for k, v in r["static"].items()})


if __name__ == "__main__":
    main()
