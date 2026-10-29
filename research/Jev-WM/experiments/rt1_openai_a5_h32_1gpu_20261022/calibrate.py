"""Small Decisions pilot calibration using existing SFT training-window rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
CAL = ROOT.parent / "rt1_qwen_judge_calibration_20261019"
sys.path.insert(0, str(CAL))
sys.path.insert(0, str(ROOT / "source/src"))
from calibrate_axes import controls
from futureworlds.jev_judge import atomic_json
from futureworlds.openai_judge import OpenAIDecisionsJudge, PROTOCOL
from futureworlds.qwen_axes import score_axes
from runtime import api_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "train.json").read_text())["openai_judge"]
    cfg.update(max_http_attempts=1024, retries=2, timeout_seconds=90)
    original = json.loads((CAL / "protocol_axes.json").read_text())
    folder = CAL / "cases/sft_train_h32_confirm"
    index = json.loads((folder / "index.json").read_text())
    selected = list(range(args.cases))
    if not 2 <= args.cases <= len(index):
        raise ValueError("Select at least two existing training windows")
    protocol = dict(protocol=PROTOCOL, model=cfg["model"], scale=cfg["scale"],
                    weights=cfg["weights"], cases_folder=str(folder), selected=selected,
                    controls={a: original["controls"][a] for a in cfg["weights"]},
                    tau_rule="2 * median abs(view_gap - within-window mean view_gap)",
                    gate="All four fixed axes pass their original controls and informative rate >= .25",
                    purpose="Exploratory pilot; four train windows are not a full judge validation",
                    source={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [ROOT / "source/src/futureworlds" / n for n in
                             ("openai_judge.py", "qwen_axes.py", "qwen_judge.py")]})
    atomic_json(ROOT / "calibration/protocol.json", protocol)
    api_environment()
    judge = OpenAIDecisionsJudge(cfg)
    needed = set(selected)
    for i in selected:
        needed.add(next(j for j, item in enumerate(index) if item["task"] != index[i]["task"]))
    cases = {}
    for i in sorted(needed):
        with np.load(folder / f"{i:03d}.npz") as z:
            cases[i] = {k: z[k] for k in z.files}
    if args.probe:
        c = cases[0]
        result = score_axes(judge, c["initial"], c["reference_raw"],
                            np.stack([c["reference_codec"], np.zeros_like(c["reference_codec"])]),
                            str(c["task"]), axes=["fidelity"])["fidelity"]
        out = dict(score=result["score"].tolist(), views=result["views"].tolist())
        atomic_json(ROOT / "calibration/api_probe.json", out)
        print(json.dumps(out), flush=True)
        return
    rows = []
    started = time.monotonic()
    for i in selected:
        c = cases[i]
        task = str(c["task"])
        other = next(x for x in cases.values() if str(x["task"]) != task)
        ctrl = controls(c, other)
        clips = {f"sft{k}": p for k, p in enumerate(c["predictions"])}
        clips.update(gt_codec=c["reference_codec"], **ctrl)
        row = dict(index=i, id=index[i]["id"], task=task, axes={})
        for axis in cfg["weights"]:
            names = [f"sft{k}" for k in range(4)] + ["gt_codec"]
            names += [n for n in original["controls"][axis] if n != "wrong_task"]
            predictions = np.stack([clips[n] for n in names])
            tasks = None
            if axis == "task":
                names.append("wrong_task")
                predictions = np.concatenate([predictions, c["reference_codec"][None]])
                tasks = [task] * (len(names) - 1) + [str(other["task"])]
            result = score_axes(judge, c["initial"], c["reference_raw"], predictions, task,
                                axes=[axis], tasks=tasks)[axis]
            row["axes"][axis] = {name: dict(score=float(result["score"][k]), views=result["views"][k].tolist())
                                 for k, name in enumerate(names)}
            atomic_json(ROOT / "calibration/partial.json", dict(rows=rows, current=row))
            print(json.dumps(dict(case=i, axis=axis, seconds=round(time.monotonic() - started, 1))), flush=True)
        rows.append(row)
        atomic_json(ROOT / "calibration/scores.json", rows)
    axes = {}
    for axis in cfg["weights"]:
        res = [r["axes"][axis] for r in rows]
        scores = np.array([[r[f"sft{k}"]["score"] for k in range(4)] for r in res])
        views = np.array([[r[f"sft{k}"]["views"] for k in range(4)] for r in res])
        gap = views[..., 0] - views[..., 1]
        gap -= gap.mean(1, keepdims=True)
        tau = 2 * float(np.median(np.abs(gap)))
        gates = {}
        for name, required in original["controls"][axis].items():
            rate = float(np.mean([r["gt_codec"]["score"] > r[name]["score"] for r in res]))
            gates[name] = dict(rate=rate, required=required, passed=rate >= required)
        informative = float(np.mean(np.ptp(scores, axis=1) > tau))
        axes[axis] = dict(controls=gates, tau=tau, informative=informative,
                          included=all(x["passed"] for x in gates.values()) and informative >= .25)
    summary = dict(protocol=PROTOCOL, model=cfg["model"], scale=cfg["scale"], weights=cfg["weights"],
                   axes=axes, cases=selected, launchable=all(x["included"] for x in axes.values()),
                   seconds=time.monotonic() - started, exploratory=True)
    atomic_json(ROOT / "calibration/summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["launchable"]:
        raise RuntimeError("Pilot calibration failed; no optimizer update was started")


if __name__ == "__main__":
    main()
