"""Calibrate the five axes on 32-frame SFT candidates; gates from protocol_axes.json."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageFilter

from qwen_axes import AXES, score_axes
from qwen_judge import QwenJudge

ROOT = Path(__file__).resolve().parent
# Extra controls scored for diagnosis only (added after dev showed smooth rewards
# flicker): checks whether the GT-referenced axes can be gamed by flicker.
DIAGNOSTIC = {"fidelity": ["flicker"], "motion": ["flicker"]}


def hotspot(gt):
    diff = np.abs(gt[-1].astype(np.float32) - gt[0].astype(np.float32)).mean(-1)
    k = 64
    best, yx = -1, (0, 0)
    for y in range(0, gt.shape[1] - k + 1, 16):
        for x in range(0, gt.shape[2] - k + 1, 16):
            v = diff[y:y + k, x:x + k].sum()
            if v > best:
                best, yx = v, (y, x)
    return yx, k


def controls(case, other):
    """Synthetic failures built from the codec-decoded GT (same codec look as predictions)."""
    g, init = case["reference_codec"], case["initial"]
    T = len(g)
    (y, x), k = hotspot(g)
    dup = g.copy()
    # Duplicate: paste the moving region of frame T/2 at a displaced location from T/2 on.
    patch = g[T // 2, y:y + k, x:x + k]
    dx = (x + 120) % (g.shape[2] - k)
    dup[T // 2:, y:y + k, dx:dx + k] = patch
    erase = g.copy()
    for t in range(T // 2, T):
        region = Image.fromarray(g[t, y:y + k, x:x + k]).filter(ImageFilter.BoxBlur(24))
        erase[t, y:y + k, x:x + k] = np.asarray(region)
    q = T // 4
    splice = np.concatenate([g[:q], g[2 * q:3 * q], g[q:2 * q], g[3 * q:]])
    flicker = g.copy()
    flicker[1::2] = g[np.minimum(np.arange(1, T, 2) + 8, T - 1)]
    blur = g.copy()
    for t in range(T // 2, T):
        radius = 1 + 5 * (t - T // 2) / (T - T // 2)
        blur[t] = np.asarray(Image.fromarray(g[t]).filter(ImageFilter.GaussianBlur(radius)))
    half = np.concatenate([g[: T // 2], np.repeat(g[T // 2 - 1:T // 2], T - T // 2, 0)])
    return dict(
        freeze_all=np.repeat(init[-1:], T, 0), freeze_second_half=half, black=np.zeros_like(g),
        other_episode=other["reference_codec"], reverse=g[::-1].copy(), duplicate=dup, erase=erase,
        splice=splice, flicker=flicker, blur=blur,
    )


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    # Amendment 2026-10-20 (post-hoc, after confirm): "raw" counted a constant
    # presentation-order offset as noise; "centered" removes each window's mean
    # view gap first, since an offset shared by the 4 candidates cannot reorder them.
    parser.add_argument("--tau-rule", choices=["raw", "centered"], default="centered")
    parser.add_argument("--from-scores", default=None, help="re-summarize an existing run's scores.json; no judge calls")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol_axes.json").read_text())
    folder = ROOT / "cases" / args.cases
    index = json.loads((folder / "index.json").read_text())
    cases = []
    for r in index:
        with np.load(folder / f"{r['index']:03d}.npz") as z:
            cases.append({k: z[k] for k in z.files})
    selected = list(range(args.start, len(cases) if args.end is None else min(args.end, len(cases))))
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    tasks = [str(c["task"]) for c in cases]
    rows, started = [], time.monotonic()
    if args.from_scores:
        rows = json.loads((ROOT / "results" / args.from_scores / "scores.json").read_text())
        assert [r["index"] for r in rows] == selected and all(r["id"] == index[r["index"]]["id"] for r in rows)
    else:
        torch.cuda.set_per_process_memory_fraction(0.33, 0)
        judge = QwenJudge(args.model, "cuda:0", scale=args.scale, batch_size=args.batch_size)
    for i in ([] if args.from_scores else selected):
        case, task = cases[i], tasks[i]
        # Next window whose instruction differs, for the wrong-task control.
        j = next(j for j in list(range(i + 1, len(cases))) + list(range(i)) if tasks[j] != task)
        ctrl = controls(case, cases[j])
        row = dict(index=i, id=index[i]["id"], task=task, wrong_task=tasks[j], axes={})
        for axis in AXES:
            names = [f"sft{k}" for k in range(4)] + ["gt_codec"] + list(protocol["controls"][axis]) + DIAGNOSTIC.get(axis, [])
            names = [n for n in names if n != "wrong_task"]
            clips = {f"sft{k}": p for k, p in enumerate(case["predictions"])}
            clips.update(gt_codec=case["reference_codec"], **ctrl)
            batch = np.stack([clips[n] for n in names])
            per_task = None
            if axis == "task":
                names.append("wrong_task")
                batch = np.concatenate([batch, case["reference_codec"][None]])
                per_task = [task] * (len(names) - 1) + [tasks[j]]
            s = score_axes(judge, case["initial"], case["reference_raw"], batch, task, axes=[axis], tasks=per_task)[axis]
            row["axes"][axis] = {n: dict(score=float(s["score"][m]), views=s["views"][m].tolist()) for m, n in enumerate(names)}
        rows.append(row)
        print(json.dumps(dict(i=i, seconds=round(time.monotonic() - started, 1),
                              **{a: {n: round(v["score"], 2) for n, v in row["axes"][a].items()} for a in AXES})), flush=True)
        (out / "scores.json").write_text(json.dumps(rows) + "\n")

    report = {}
    for axis in AXES:
        res = {a: [r["axes"][axis] for r in rows] for a in [axis]}[axis]
        gates = {}
        for ctrl_name, need in protocol["controls"][axis].items():
            rate = float(np.mean([r["gt_codec"]["score"] > r[ctrl_name]["score"] for r in res]))
            gates[ctrl_name] = dict(rate=rate, required=need, passed=rate >= need)
        sft = np.array([[r[f"sft{k}"]["score"] for k in range(4)] for r in res])
        views = np.array([[r[f"sft{k}"]["views"] for k in range(4)] for r in res])
        gap = views[..., 0] - views[..., 1]
        if args.tau_rule == "centered":
            gap = gap - gap.mean(1, keepdims=True)
        tau = 2 * float(np.median(np.abs(gap)))
        informative = float(np.mean(sft.max(1) - sft.min(1) > tau))
        lp = np.array([index[i]["lpips"] for i in selected])
        reps = np.array([index[i]["frame_repeats"] for i in selected])
        diagnostic = {c: float(np.mean([r["gt_codec"]["score"] > r[c]["score"] for r in res])) for c in DIAGNOSTIC.get(axis, []) if c in res[0]}
        report[axis] = dict(
            controls=gates, diagnostic_gt_beats=diagnostic, tau=tau, informative=informative,
            mean_view_gap=float((views[..., 0] - views[..., 1]).mean()),
            included=all(g["passed"] for g in gates.values()) and informative >= 0.25,
            mean_scores={n: float(np.mean([r[n]["score"] for r in res])) for n in res[0]},
            median_group_spread=float(np.median(sft.max(1) - sft.min(1))),
            within_group_spearman_views=float(np.nanmean([spearman(v[:, 0], v[:, 1]) for v in views])),
            within_group_spearman_vs_neg_lpips=float(np.nanmean([spearman(s, -l) for s, l in zip(sft, lp)])),
            within_group_spearman_vs_neg_repeats=float(np.nanmean([spearman(s, -r) for s, r in zip(sft, reps)])),
        )
    cross = {}
    for a in AXES:
        for b in AXES:
            if a < b:
                sa = np.array([[r["axes"][a][f"sft{k}"]["score"] for k in range(4)] for r in rows])
                sb = np.array([[r["axes"][b][f"sft{k}"]["score"] for k in range(4)] for r in rows])
                cross[f"{a}~{b}"] = float(np.nanmean([spearman(x, y) for x, y in zip(sa, sb)]))
    arms = {}
    for arm in ["arm_FM", "arm_A5"]:
        w = {a: v for a, v in protocol[arm]["weights"].items() if report[a]["included"]}
        ok = report["fidelity"]["included"] and report["motion"]["included"]
        total = sum(w.values())
        arms[arm] = dict(launchable=ok, weights={a: v / total for a, v in w.items()} if ok else None,
                         taus={a: report[a]["tau"] for a in w})
    summary = dict(model=args.model, scale=args.scale, cases_folder=args.cases, cases=selected, axes=report,
                   tau_rule=args.tau_rule, scores_from=args.from_scores or args.name,
                   within_group_axis_spearman=cross, arms=arms, seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
