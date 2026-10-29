"""Interim SFT vs RL checkpoint comparison on identical windows (image metrics + exact stall)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent / "rt1_memspo200_upstream_20261017"


def stall(folder):
    """Per case: fraction of the 31 future transitions whose frame tokens equal the previous frame's."""
    out = []
    for case in sorted(folder.glob("case_*")):
        with np.load(case / "prediction.npz") as z:
            p = z["tokens"][0]  # [32, 80] visual tokens per future frame
        # Identical tokens decode to identical frames (token and RGB repeats match one-to-one).
        same = [np.array_equal(a, b) for a, b in zip(p[1:], p[:-1])]
        runs, cur = [], 0
        for s in same:
            cur = cur + 1 if s else 0
            runs.append(cur)
        out.append((float(np.mean(same)), max(runs) if runs else 0, [float(np.mean(same[i:i + 8])) for i in range(0, 31, 8)]))
    return out


def boot(delta, seed=20261020):
    rng = np.random.default_rng(seed)
    return np.quantile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1), [0.025, 0.975])


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    runs = {"sft": ROOT.parent / f"evaluation/{split}/sft", "rl": ROOT.parent / f"evaluation/{split}/rl"}
    summ = {k: json.loads((v / "summary.json").read_text()) for k, v in runs.items()}
    ids = [x["id"] for x in summ["sft"]["per_case"]]
    assert [x["id"] for x in summ["rl"]["per_case"]] == ids
    for k in ["manifest_sha256", "horizons", "beam_width", "memory", "precision", "tf32", "initial_observations", "gt_refresh"]:
        assert summ["rl"]["protocol"][k] == summ["sft"]["protocol"][k], k
    lines = [f"# 中期对比：SFT vs Qwen 四维 RL step100（{split}，{len(ids)} 窗口，同一批窗口）", "",
             "|帧数|指标|SFT|RL step100|RL−SFT 95% CI|改善窗口数|", "|---:|---|---:|---:|---|---:|"]
    result = {}
    for h in ["10", "20", "32"]:
        for m in ["lpips", "psnr", "ssim", "mae"]:
            a = {k: np.array([x["metrics"][h][m] for x in s["per_case"]]) for k, s in summ.items()}
            sign = -1 if m in ("lpips", "mae") else 1
            d = a["rl"] - a["sft"]
            ci = boot(d)
            result[f"{h}/{m}"] = dict(sft=float(a["sft"].mean()), rl=float(a["rl"].mean()), ci=ci.tolist())
            lines.append(f"|{h}|{m}|{a['sft'].mean():.4f}|{a['rl'].mean():.4f}|[{ci[0]:+.4f}, {ci[1]:+.4f}]|{int((sign * d > 0).sum())}/{len(d)}|")
    st = {k: stall(v) for k, v in runs.items()}
    frac = {k: np.array([x[0] for x in v]) for k, v in st.items()}
    longest = {k: np.array([x[1] for x in v]) for k, v in st.items()}
    seg = {k: np.array([x[2] for x in v]) for k, v in st.items()}
    ci = boot(frac["rl"] - frac["sft"])
    lines += ["", "## 停滞（32 帧预测，相邻帧视觉 token 完全相同）", "",
              "|指标|SFT|RL step100|", "|---|---:|---:|",
              f"|静止帧比例|{frac['sft'].mean():.3f}|{frac['rl'].mean():.3f}|",
              f"|RL−SFT 95% CI|—|[{ci[0]:+.3f}, {ci[1]:+.3f}]|",
              f"|最长连续静止 ≥10 帧的窗口|{int((longest['sft'] >= 10).sum())}/{len(ids)}|{int((longest['rl'] >= 10).sum())}/{len(ids)}|",
              f"|静止更少的窗口|—|{int((frac['rl'] < frac['sft']).sum())}/{len(ids)}（更多 {int((frac['rl'] > frac['sft']).sum())}）|"]
    for i, name in enumerate(["1–8", "9–16", "17–24", "25–31"]):
        lines.append(f"|帧间 {name} 静止比例|{seg['sft'][:, i].mean():.3f}|{seg['rl'][:, i].mean():.3f}|")
    result["stall"] = dict(sft=float(frac["sft"].mean()), rl=float(frac["rl"].mean()), ci=ci.tolist())
    lines += ["", "固定最终 step100，单种子；SFT 与 RL 在本实验内用同一评测代码重新评测。验证集前32窗口曾用于早期10帧裁判校准；测试集此前只用于查看 R0 结果。"]
    (ROOT / f"REPORT_{split}.md").write_text("\n".join(lines) + "\n")
    (ROOT / f"compare_{split}.json").write_text(json.dumps(result, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
