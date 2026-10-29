"""One table: pretrained base / FutureWorlds SFT / R0 RL / Qwen FM / Qwen A5, identical windows and protocol."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

E = Path("/media/tyshao/Jev-WM/experiments")
RUNS = {
    "预训练底座（无SFT无RL）": E / "rt1_pretrained_base_eval_20261021/base/{s}",
    "FutureWorlds SFT": E / "rt1_qwen_a5_h32_20261020/evaluation/{s}/sft",
    "R0 RL（h10, 200步，参考）": E / "rt1_memspo200_upstream_20261017/evaluation/{s}/rl",
    "Qwen 二维 RL": E / "rt1_qwen_fm_h32_20261020/evaluation/{s}/rl",
    "Qwen 四维 RL": E / "rt1_qwen_a5_h32_20261020/evaluation/{s}/rl",
}


def stall(folder):
    frac, longest = [], []
    for c in sorted(glob.glob(str(folder / "case_*"))):
        p = np.load(c + "/prediction.npz")["tokens"][0]
        same = [np.array_equal(a, b) for a, b in zip(p[1:], p[:-1])]
        run = cur = 0
        for x in same:
            cur = cur + 1 if x else 0
            run = max(run, cur)
        frac.append(np.mean(same))
        longest.append(run)
    return np.array(frac), np.array(longest)


def boot(d, seed=20261021):
    rng = np.random.default_rng(seed)
    return np.quantile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [0.025, 0.975])


def main():
    lines = ["# RT-1 世界模型：预训练底座 vs SFT vs RL（同一批窗口、同一 ReCAP 推理协议）", "",
             "推理：2 帧初始观测，ReCAP（anchor1 + recent6），逐帧 beam4，无 GT 刷新，自主预测 32 帧。",
             "预训练底座 = thuml/rt1-world-model-multi-step-base @242d6b7（RLVR-World），无文本条件；其余模型带 T5 文本适配器。", ""]
    out = {}
    for split in sys.argv[1:] or ["test", "validation"]:
        summ = {k: json.loads((Path(str(v).format(s=split)) / "summary.json").read_text()) for k, v in RUNS.items()}
        ref = list(summ)[1]
        ids = [x["id"] for x in summ[ref]["per_case"]]
        for k, s in summ.items():
            assert [x["id"] for x in s["per_case"]] == ids, k
            for f in ["manifest_sha256", "horizons", "beam_width", "memory", "precision", "tf32", "initial_observations", "gt_refresh"]:
                assert s["protocol"][f] == summ[ref]["protocol"][f], (k, f)
        lines += [f"## {split}（{len(ids)} 窗口）", "",
                  "|模型|LPIPS@10|LPIPS@20|LPIPS@32|PSNR@32|SSIM@32|静止帧比例|冻结≥10帧窗口|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        res = {}
        for k, s in summ.items():
            m = lambda h, q: np.array([x["metrics"][h][q] for x in s["per_case"]])  # noqa: E731
            fr, lo = stall(Path(str(RUNS[k]).format(s=split)))
            res[k] = dict(lp10=m("10", "lpips"), lp20=m("20", "lpips"), lp32=m("32", "lpips"), ps32=m("32", "psnr"),
                          ss32=m("32", "ssim"), static=fr, longest=lo)
            r = res[k]
            lines.append(f"|{k}|{r['lp10'].mean():.4f}|{r['lp20'].mean():.4f}|{r['lp32'].mean():.4f}|{r['ps32'].mean():.2f}|"
                         f"{r['ss32'].mean():.4f}|{r['static'].mean():.1%}|{int((r['longest'] >= 10).sum())}/{len(ids)}|")
        base = list(res)[0]
        lines += ["", "相对预训练底座的配对差（95% bootstrap CI）：", "",
                  "|模型|ΔLPIPS@32|ΔPSNR@32|Δ静止帧比例|", "|---|---|---|---|"]
        for k in list(res)[1:]:
            c1, c2, c3 = (boot(res[k][q] - res[base][q]) for q in ["lp32", "ps32", "static"])
            lines.append(f"|{k}|{(res[k]['lp32'] - res[base]['lp32']).mean():+.4f} [{c1[0]:+.4f}, {c1[1]:+.4f}]|"
                         f"{(res[k]['ps32'] - res[base]['ps32']).mean():+.2f} [{c2[0]:+.2f}, {c2[1]:+.2f}]|"
                         f"{(res[k]['static'] - res[base]['static']).mean():+.1%} [{c3[0]:+.1%}, {c3[1]:+.1%}]|")
        lines.append("")
        out[split] = {k: {q: float(v.mean()) for q, v in r.items()} for k, r in res.items()}
    lines += ["单种子；验证集前32窗口曾用于早期裁判校准；测试集此前用于查看 R0 结果。R0 RL 训练帧数与步数不同，仅作参考。",
              "预训练底座与 FutureWorlds SFT 的训练数据划分均未公开，与本地测试窗口的重合情况未知。"]
    dest = E / "rt1_pretrained_base_eval_20261021"
    (dest / "REPORT.md").write_text("\n".join(lines) + "\n")
    (dest / "comparison.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
