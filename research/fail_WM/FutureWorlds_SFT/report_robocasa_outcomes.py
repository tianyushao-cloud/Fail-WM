#!/usr/bin/env python3
"""Report outcome-aware RoboCasa world-model evaluation metrics.

The FutureWorlds evaluator scores predicted pixels against the audited future
frames. This report joins those scores with held-out success/failure metadata.
It calibrates a failure-risk threshold on validation only, then reports the
test failure-to-success error rate (the operational optimism rate).
"""

import argparse
import json
import math
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text())


def load_labels(path):
    manifest = load_json(path)
    rows = manifest.get("cases", [])
    labels = {}
    for row in rows:
        if row["id"] in labels:
            raise ValueError(f"Duplicate manifest id: {row['id']}")
        if "success" not in row or not isinstance(row["success"], bool):
            raise ValueError(f"Missing boolean success label: {row['id']}")
        labels[row["id"]] = {
            "success": row["success"],
            "failure_type": row.get("failure_type") or "unknown_failure",
            "task": row.get("task", "unknown"),
            "initial_state_id": row.get("initial_state_id", "unknown"),
        }
    if not labels:
        raise ValueError(f"Empty manifest: {path}")
    return labels


def load_summary(path):
    value = load_json(path)
    cases = value.get("per_case")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Summary has no per_case results: {path}")
    return value


def joined(summary, labels, horizon, metric):
    result = []
    seen = set()
    for case in summary["per_case"]:
        cid = case.get("id")
        if cid not in labels:
            raise ValueError(f"Result id is absent from manifest: {cid}")
        if cid in seen:
            raise ValueError(f"Duplicate result id: {cid}")
        seen.add(cid)
        value = case.get("metrics", {}).get(str(horizon), {}).get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Missing finite {metric} at horizon {horizon}: {cid}")
        result.append({**labels[cid], "id": cid, "score": float(value)})
    if set(seen) != set(labels):
        missing = sorted(set(labels) - seen)
        raise ValueError(f"Manifest/result cohort mismatch; missing {missing[:5]}")
    return result


def mean(values):
    return sum(values) / len(values) if values else None


def auc(rows):
    positives = [r["score"] for r in rows if not r["success"]]
    negatives = [r["score"] for r in rows if r["success"]]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def confusion(rows, threshold):
    tp = fp = tn = fn = 0
    for row in rows:
        predicted_failure = row["score"] >= threshold
        actual_failure = not row["success"]
        if predicted_failure and actual_failure:
            tp += 1
        elif predicted_failure and not actual_failure:
            fp += 1
        elif not predicted_failure and not actual_failure:
            tn += 1
        else:
            fn += 1
    n_pos = tp + fn
    n_neg = tn + fp
    tpr = tp / n_pos if n_pos else None
    tnr = tn / n_neg if n_neg else None
    ba = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    return {
        "tp_failure": tp,
        "fn_failure_predicted_success": fn,
        "tn_success": tn,
        "fp_success_predicted_failure": fp,
        "sensitivity": tpr,
        "specificity": tnr,
        "balanced_accuracy": ba,
        "optimism_rate": fn / n_pos if n_pos else None,
    }


def calibrate(rows):
    values = sorted({r["score"] for r in rows})
    candidates = [values[0] - 1e-12]
    candidates.extend((a + b) / 2 for a, b in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-12)
    best = None
    for threshold in candidates:
        stats = confusion(rows, threshold)
        key = (
            -1.0 if stats["balanced_accuracy"] is None else stats["balanced_accuracy"],
            -stats["fp_success_predicted_failure"],
            -threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, stats)
    return {"threshold": best[1], **best[2]}


def grouped(rows, horizon, metric):
    groups = {"all": rows, "success": [r for r in rows if r["success"]], "failure": [r for r in rows if not r["success"]]}
    failure_types = sorted({r["failure_type"] for r in rows if not r["success"]})
    groups.update({f"failure:{name}": [r for r in rows if not r["success"] and r["failure_type"] == name] for name in failure_types})
    output = {}
    for name, values in groups.items():
        output[name] = {"n": len(values), "mean": mean([r["score"] for r in values])}
    return output


def build_model_report(name, val_summary, test_summary, val_labels, test_labels, horizon, metric):
    val_rows = joined(val_summary, val_labels, horizon, metric)
    test_rows = joined(test_summary, test_labels, horizon, metric)
    calibration = calibrate(val_rows)
    test_calibrated = confusion(test_rows, calibration["threshold"])
    return {
        "name": name,
        "metric": metric,
        "horizon": horizon,
        "validation": {
            "n": len(val_rows),
            "groups": grouped(val_rows, horizon, metric),
            "failure_auc": auc(val_rows),
            "calibration": calibration,
        },
        "test": {
            "n": len(test_rows),
            "groups": grouped(test_rows, horizon, metric),
            "failure_auc": auc(test_rows),
            "calibrated_at_validation_threshold": test_calibrated,
        },
    }


def markdown(report):
    lines = [
        "# RoboCasa Outcome Evaluation",
        "",
        f"Metric: `{report['metric']}` at horizon `{report['horizon']}`.",
        "The threshold is selected on validation only. `optimism_rate` is the fraction of actual failures classified as success by that threshold.",
        "",
        "| run | test success error | test failure error | failure AUROC | optimism rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["models"]:
        test = item["test"]
        groups = test["groups"]
        calibrated = test["calibrated_at_validation_threshold"]
        lines.append(
            f"| {item['name']} | {groups['success']['mean']:.6f} | {groups['failure']['mean']:.6f} | "
            f"{test['failure_auc']:.4f} | {calibrated['optimism_rate']:.4f} |"
        )
    lines.extend(["", "Failure-type breakdown:", "", "| run | failure type | n | mean error |", "|---|---|---:|---:|"])
    for item in report["models"]:
        for key, value in item["test"]["groups"].items():
            if key.startswith("failure:"):
                lines.append(f"| {item['name']} | {key[8:]} | {value['n']} | {value['mean']:.6f} |")
    return "\n".join(lines) + "\n"


def parse_run(value):
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("run must be NAME:VAL_SUMMARY:TEST_SUMMARY")
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--run", action="append", required=True, type=parse_run, metavar="NAME:VAL:TEST")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--metric", choices=["lpips", "mae", "mse"], default="lpips")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    val_labels = load_labels(args.val_manifest)
    test_labels = load_labels(args.test_manifest)
    models = [build_model_report(name, load_summary(val), load_summary(test), val_labels, test_labels, args.horizon, args.metric) for name, val, test in args.run]
    report = {"state": "COMPLETE", "val_manifest": str(Path(args.val_manifest).resolve()), "test_manifest": str(Path(args.test_manifest).resolve()), "horizon": args.horizon, "metric": args.metric, "models": models}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"json": str(output.with_suffix('.json')), "markdown": str(output.with_suffix('.md'))}, indent=2))


if __name__ == "__main__":
    main()
