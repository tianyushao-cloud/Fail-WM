import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "rt1_memspo200_upstream_20261017"


def read(p):
    return json.loads(Path(p).read_text())


def main():
    cfg = read(ROOT / "train.json")
    rows = [json.loads(x) for x in (ROOT / "train/metrics.jsonl").read_text().splitlines()]
    # Resumes can replay rows; keep the last committed occurrence of each batch.
    steps = {r["step"]: r for r in rows}
    assert sorted(steps) == list(range(1, cfg["steps"] + 1))
    keys = sorted({k for r in steps.values() for k in r["judge"] if k.startswith(("judged", "informative"))})
    totals = {k: sum(r["judge"].get(k, 0) for r in steps.values()) for k in keys}
    updates = sum(r["optimizer_updated"] for r in steps.values())
    upstream = read(ROOT / "upstream_source_hashes.json")
    for name, h in upstream.items():
        if name != "src/futureworlds/training.py":
            assert hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest() == h, name
    result = dict(judge=cfg["openai_judge"], scheduled_batches=cfg["steps"], optimizer_updates=updates, totals=totals, splits={})
    qj = cfg["openai_judge"]
    migration = read(ROOT / "migration.json")
    result["continuation"] = migration
    lines = [f"# RT-1：SFT vs 纯 OpenAI Decisions 多维奖励 MemSPO{cfg['steps']}（pointwise 校准后维度，训练 horizon {cfg['horizon']}）", "",
             "第1–10步继承四卡实验，第11–14步继承八卡16并发实验，第15–100步为八卡最多256并发续训；actor、Adam和全部8个rank的RNG由第14步完整迁移。",
             "八卡版本重新加载原始RT-1 SFT参考模型，修正deepcopy文本hook绑定actor的问题；卡数与此修正使它不与旧四卡实现逐位等价。",
             f"global batch {cfg['global_batch']}，训练预测{cfg['horizon']}帧。{cfg['steps']}个计划批次，实际optimizer更新{updates}次。"
             "奖励只来自OpenAI Decisions 裁判，无R0/LPIPS/MAE奖励或筛选。",
             f"裁判：{qj['model']}，scale={qj['scale']}；维度权重 {qj['weights']}；"
             f"有效组 {totals['informative_groups']:.0f}/{totals['judged_groups']:.0f}；各维度有效次数 "
             + ", ".join(f"{k[12:]} {v:.0f}" for k, v in totals.items() if k.startswith("informative_") ) + "。",
             "GT 只在训练奖励端可见；生成和评测不注入GT。固定最终checkpoint，不按测试结果选择。R0 RL 列仅作参考，不是本实验目标。", "",
             "|划分|帧数|指标|SFT|Decisions RL|Decisions RL−SFT 95% CI|相对SFT|R0 RL（参考）|",
             "|---|---:|---|---:|---:|---|---:|---:|"]
    for split in ["validation", "test"]:
        runs = {name: read(path) for name, path in [("sft", ROOT / f"evaluation/{split}/sft/summary.json"),
                                                    ("openai", ROOT / f"evaluation/{split}/rl/summary.json")]}
        r0_path = BASE / f"evaluation/{split}/rl/summary.json"
        r0 = read(r0_path) if r0_path.exists() else None
        ids = [x["id"] for x in runs["sft"]["per_case"]]
        for r in list(runs.values()) + ([r0] if r0 else []):
            assert [x["id"] for x in r["per_case"]] == ids
            for k in ["manifest_sha256", "horizons", "beam_width", "memory", "precision", "tf32", "initial_observations", "gt_refresh", "text_assets"]:
                assert r["protocol"][k] == runs["sft"]["protocol"][k], k
        split_result = {}
        for h in ["10", "20", "32"]:
            values = {}
            for metric in ["lpips", "psnr", "ssim", "mae", "mse"]:
                a = {n: np.array([x["metrics"][h][metric] for x in r["per_case"]]) for n, r in runs.items()}
                sign = -1 if metric in ["lpips", "mae", "mse"] else 1
                delta = a["openai"] - a["sft"]
                rng = np.random.default_rng(20261019)
                ci = np.quantile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1), [0.025, 0.975])
                mean_sft = float(a["sft"].mean())
                rel = 100 * sign * float(delta.mean()) / mean_sft
                r0_mean = float(np.mean([x["metrics"][h][metric] for x in r0["per_case"]])) if r0 else None
                values[metric] = dict(sft=mean_sft, openai=float(a["openai"].mean()), r0_reference=r0_mean, delta=float(delta.mean()),
                                      paired_episode_bootstrap_95ci=ci.tolist(), relative_improvement=rel,
                                      improved_cases=int((sign * delta > 0).sum()), cases=len(ids))
                shown = f"{delta.mean():+.3f} dB" if metric == "psnr" else f"{rel:+.2f}%"
                r0_shown = "—" if r0_mean is None else f"{r0_mean:.6f}"
                lines.append(f"|{split}|{h}|{metric}|{mean_sft:.6f}|{a['openai'].mean():.6f}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|{shown}|{r0_shown}|")
            split_result[h] = values
        result["splits"][split] = split_result
    lines += ["", "一次训练种子、本地数据划分；校准复用4个train窗口；验证和测试划分沿用既有探索实验，不是新盲测。",
              "图像指标不等于任务成功率；CI 跨 0 时不声称稳健提升。"]
    selection = read(cfg["openai_judge"]["calibration_file"])
    result["judge_calibration"] = selection
    if selection.get("selection_amendment"):
        lines += ["", "裁判校准：physics 的 erase 控制通过率 0.50，未达 0.75 门槛。完整校准后，按原 Qwen A5 的失败轴剔除规则，保留 fidelity/motion/task 并重新归一化权重；原四维失败记录已保留。",
                  "这是完整校准后对首次整体门槛的明确修订。与原 Qwen 四维结果比较，同时改变裁判和纳入维度，不构成单变量消融。"]
    interim_path = ROOT / f"evaluation/validation/step{cfg.get('interim_step', cfg['checkpoint_every'])}/summary.json"
    if interim_path.exists():
        interim = read(interim_path)
        sft = read(ROOT / "evaluation/validation/sft/summary.json")
        final = read(ROOT / "evaluation/validation/rl/summary.json")
        ids = [r["id"] for r in sft["per_case"]]
        assert [r["id"] for r in interim["per_case"]] == ids
        for key in ["manifest_sha256", "horizons", "beam_width", "memory", "precision", "tf32", "initial_observations", "gt_refresh", "text_assets"]:
            assert interim["protocol"][key] == sft["protocol"][key], key
        lines += ["", "## Validation 趋势", "",
                  f"|预测帧数|指标|SFT|第{cfg['interim_step']}步|第{cfg['steps']}步|", "|---|---|---:|---:|---:|"]
        trend = {}
        for h in ["10", "20", "32"]:
            trend[h] = {}
            for metric in ["lpips", "psnr", "ssim", "mae", "mse"]:
                values = [float(np.mean([r["metrics"][h][metric] for r in run["per_case"]]))
                          for run in [sft, interim, final]]
                trend[h][metric] = dict(zip(["sft", f"step{cfg['interim_step']}", f"step{cfg['steps']}"], values))
                lines.append(f"|{h}|{metric}|{values[0]:.6f}|{values[1]:.6f}|{values[2]:.6f}|")
        result["validation_trend"] = trend
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
