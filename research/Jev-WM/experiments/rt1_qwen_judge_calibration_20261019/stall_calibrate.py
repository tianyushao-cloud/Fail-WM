"""Calibrate the motion-strip stall question against EXACT stall labels.

Exact label per segment: fraction of transitions where the prediction frame is
byte-identical to the previous one while the reference visibly moved. Labels
are only used to measure the judge; they are never a reward.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from qwen_judge import QwenJudge, stall_scores

ROOT = Path(__file__).resolve().parent
MOVE = 1.0  # mean |diff| (0-255) above which a reference transition counts as motion


def exact_stall(initial, reference, clip, segments):
    prev_c = np.concatenate([initial[-1:], clip[:-1]])
    prev_r = np.concatenate([initial[-1:], reference[:-1]])
    static = np.array([np.array_equal(a, b) for a, b in zip(clip, prev_c)])
    moved = np.abs(reference.astype(np.float32) - prev_r.astype(np.float32)).mean((1, 2, 3)) > MOVE
    stall = static & moved
    return [float(stall[s:e].mean()) for s, e in segments]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cases", default="sft_train_h32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    folder = ROOT / "cases" / args.cases
    index = json.loads((folder / "index.json").read_text())
    selected = list(range(args.start, len(index) if args.end is None else min(args.end, len(index))))
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.33, 0)
    judge = QwenJudge(args.model, "cuda:0", batch_size=args.batch_size)
    rows, started = [], time.monotonic()
    for i in selected:
        with np.load(folder / f"{i:03d}.npz") as z:
            init, ref, gtc, preds, task = z["initial"], z["reference_raw"], z["reference_codec"], z["predictions"], str(z["task"])
        T = len(ref)
        half = np.concatenate([gtc[: T // 2], np.repeat(gtc[T // 2 - 1: T // 2], T - T // 2, 0)])
        clips = {f"sft{k}": p for k, p in enumerate(preds)}
        clips.update(gt_codec=gtc, freeze_all=np.repeat(init[-1:], T, 0), freeze_second_half=half)
        names = list(clips)
        s = stall_scores(judge, init, ref, np.stack([clips[k] for k in names]), task)
        row = dict(index=i, task=task, segments=s["segments"], clips={})
        for j, k in enumerate(names):
            row["clips"][k] = dict(score=s["score"][j].tolist(), per_order=s["per_order"][j].tolist(),
                                   exact=exact_stall(init, ref, clips[k], s["segments"]))
        rows.append(row)
        print(json.dumps(dict(i=i, seconds=round(time.monotonic() - started, 1),
                              **{k: [round(x, 2) for x in v["score"]] for k, v in row["clips"].items()})), flush=True)
        (out / "scores.json").write_text(json.dumps(rows) + "\n")

    seg_score, seg_exact, gap = [], [], []
    for r in rows:
        for k, v in r["clips"].items():
            seg_score += v["score"]
            seg_exact += v["exact"]
            gap += [abs(a - b) for a, b in v["per_order"]]
    seg_score, seg_exact = np.array(seg_score), np.array(seg_exact)
    nseg = len(rows[0]["segments"])
    by_pos = {}
    for p in range(nseg):
        e = [r["clips"][f"sft{k}"]["exact"][p] for r in rows for k in range(4)]
        sc = [r["clips"][f"sft{k}"]["score"][p] for r in rows for k in range(4)]
        by_pos[f"seg{p}"] = dict(sft_exact_stall_mean=float(np.mean(e)), sft_judge_score_mean=float(np.mean(sc)))
    group_rho, group_defined = [], 0
    for r in rows:
        tot_s = [np.mean(r["clips"][f"sft{k}"]["score"]) for k in range(4)]
        tot_e = [-np.mean(r["clips"][f"sft{k}"]["exact"]) for k in range(4)]
        rho = spearman(tot_s, tot_e)
        if not np.isnan(rho):
            group_rho.append(rho)
            group_defined += 1
    summary = dict(
        model=args.model, cases=selected, move_threshold=MOVE,
        segment_spearman_score_vs_neg_exact=spearman(seg_score, -seg_exact),
        auc_clean_vs_stall_ge_3of8=auc(seg_score[seg_exact == 0], seg_score[seg_exact >= 3 / 8]),
        auc_clean_vs_any_stall=auc(seg_score[seg_exact == 0], seg_score[seg_exact > 0]),
        mean_score_clean=float(seg_score[seg_exact == 0].mean()) if (seg_exact == 0).any() else None,
        mean_score_stall_ge_half=float(seg_score[seg_exact >= 0.5].mean()) if (seg_exact >= 0.5).any() else None,
        median_order_gap=float(np.median(gap)),
        within_group_spearman_total_score_vs_neg_exact=float(np.mean(group_rho)) if group_rho else None,
        groups_with_exact_variation=group_defined,
        by_segment_position=by_pos,
        controls={k: float(np.mean([np.mean(r["clips"][k]["score"]) for r in rows])) for k in ["gt_codec", "freeze_all", "freeze_second_half"]},
        seconds=time.monotonic() - started,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
