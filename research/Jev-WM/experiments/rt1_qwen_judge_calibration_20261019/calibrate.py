"""Stage 2 (judge only): score candidates and controls, evaluate preset gates.

Gates are fixed in protocol.json before any score is seen. LPIPS/MAE and the
R0-RL candidates are diagnostics only; they never select prompts or thresholds.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from qwen_judge import QwenJudge

ROOT = Path(__file__).resolve().parent


def controls(case, other):
    ref = case["reference_codec"]
    hold = np.concatenate([ref[:5], np.repeat(ref[4:5], len(ref) - 5, 0)])
    return dict(
        gt_raw=case["reference_raw"],
        gt_codec=ref,
        freeze_all=np.repeat(case["initial"][-1:], len(ref), 0),
        freeze_half=hold,
        reverse=ref[::-1].copy(),
        black=np.zeros_like(ref),
        other_episode=other["reference_codec"],
    )


def load(variant):
    folder = ROOT / "cases" / variant
    if not (folder / "index.json").exists():
        return None, None
    index = json.loads((folder / "index.json").read_text())
    cases = []
    for r in index:
        with np.load(folder / f"{r['index']:03d}.npz") as z:
            cases.append({k: z[k] for k in z.files})
    return index, cases


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    index, cases = load("sft")
    rl_index, rl_cases = load("r0rl")
    selected = list(range(args.start, len(cases) if args.end is None else min(args.end, len(cases))))
    torch.cuda.set_per_process_memory_fraction(0.33, 0)
    judge = QwenJudge(args.model, "cuda:0", scale=args.scale, max_pixels=args.max_pixels, batch_size=args.batch_size)
    rows = []
    started = time.monotonic()
    for i in selected:
        case, task = cases[i], str(cases[i]["task"])
        ctrl = controls(case, cases[(i + 1) % len(cases)])
        clips = {f"sft{k}": p for k, p in enumerate(case["predictions"])}
        if rl_cases is not None:
            assert rl_index[i]["id"] == index[i]["id"]
            clips.update({f"rl{k}": p for k, p in enumerate(rl_cases[i]["predictions"])})
        clips.update(ctrl)
        names = list(clips)
        s = judge.score(case["initial"], case["reference_raw"], np.stack([clips[k] for k in names]), task)
        row = dict(index=i, id=index[i]["id"], task=task, scores={})
        for j, k in enumerate(names):
            row["scores"][k] = dict(
                combined=float(s["combined"][j]),
                per_order=s["per_order"][j].tolist(),
                per_question={q: s["per_question"][j, :, m].tolist() for m, q in enumerate(s["questions"])},
                logits={q: s["logits"][j, :, m].tolist() for m, q in enumerate(s["questions"])},
            )
        rows.append(row)
        print(json.dumps(dict(i=i, seconds=round(time.monotonic() - started, 1),
                              **{k: round(v["combined"], 3) for k, v in row["scores"].items()})), flush=True)
        (out / "scores.json").write_text(json.dumps(rows) + "\n")

    def get(name, field="combined"):
        return np.array([r["scores"][name][field] for r in rows])

    gates = {}
    for ctrl_name, need in protocol["control_gates"].items():
        rate = float(np.mean(get("gt_codec") > get(ctrl_name)))
        gates[f"gt_codec>{ctrl_name}"] = dict(rate=rate, required=need, passed=rate >= need)
    sft = [f"sft{k}" for k in range(4)]
    po = np.array([[r["scores"][k]["per_order"] for k in sft] for r in rows])  # [n,4,2]
    combined = po.mean(-1)
    order_gap = np.abs(po[..., 0] - po[..., 1])
    tau = protocol["tie_multiplier"] * float(np.median(order_gap))
    spread = combined.max(1) - combined.min(1)
    informative = float(np.mean(spread > tau))
    gates["informative_groups"] = dict(rate=informative, required=protocol["informative_required"],
                                       passed=informative >= protocol["informative_required"], tau=tau)
    lp = np.array([index[i]["lpips"] for i in selected])
    reps = np.array([index[i]["frame_repeats"] for i in selected])
    order_rank = [spearman(p[:, 0], p[:, 1]) for p in po]
    rho_lpips = [spearman(c, -l) for c, l in zip(combined, lp)]
    rho_rep = [spearman(c, -r) for c, r in zip(combined, reps)]
    diag = dict(
        median_order_gap=float(np.median(order_gap)),
        mean_order_bias=float(np.mean(po[..., 0] - po[..., 1])),
        median_group_spread=float(np.median(spread)),
        within_group_spearman_order0_vs_order1=float(np.nanmean(order_rank)),
        order_rank_groups_defined=int(np.sum(~np.isnan(order_rank))),
        within_group_spearman_vs_neg_lpips=float(np.nanmean(rho_lpips)),
        within_group_spearman_vs_neg_repeats=float(np.nanmean(rho_rep)),
        groups_with_repeat_variation=int(np.sum([np.ptp(r) > 0 for r in reps])),
        pooled_spearman_vs_neg_lpips=spearman(combined.ravel(), -lp.ravel()),
        mean_scores={k: float(get(k).mean()) for k in rows[0]["scores"]},
        seconds=time.monotonic() - started,
    )
    if rl_cases is not None:
        rl = np.array([[r["scores"][f"rl{k}"]["combined"] for k in range(4)] for r in rows]).mean(1)
        diff = rl - combined.mean(1)
        boot = np.random.default_rng(0).choice(diff, (5000, len(diff))).mean(1)
        diag["r0rl_minus_sft_window_mean"] = dict(mean=float(diff.mean()), ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                                                  rl_better_rate=float(np.mean(diff > 0)))
    summary = dict(model=args.model, name=args.name, cases=selected, scale=args.scale, max_pixels=args.max_pixels, gates=gates,
                   all_gates_passed=all(g["passed"] for g in gates.values()), diagnostics=diag)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
