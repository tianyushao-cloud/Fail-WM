"""Apply the original Qwen A5 inclusion rule to completed Decisions calibration.

This is an explicit post-calibration amendment to the initial all-four gate.
Original scores, thresholds and failed-axis results are preserved.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    cfg_path = ROOT / "train.json"
    cfg = json.loads(cfg_path.read_text())
    judge = cfg["openai_judge"]
    path = Path(judge["calibration_file"])
    summary = json.loads(path.read_text())
    assert summary["cases"] == [0, 1, 2, 3], "Wait for all four windows"
    requested = judge.get("requested_weights", judge["weights"])
    baseline = json.loads((ROOT.parent / "rt1_qwen_a5_h32_20261020/train.json").read_text())
    assert requested == baseline["qwen_judge"]["weights"]
    original_protocol = ROOT.parent / "rt1_qwen_judge_calibration_20261019/protocol_axes.json"
    original = json.loads(original_protocol.read_text())
    assert original["arm_A5"]["rule"] == "failed axes are dropped and remaining weights renormalized; requires fidelity and motion"
    if not all(summary["axes"][axis]["included"] for axis in ["fidelity", "motion"]):
        raise RuntimeError("Required fidelity/motion axes failed; no RL is allowed")
    included = {a: w for a, w in requested.items() if summary["axes"][a]["included"]}
    selected = {a: w / sum(included.values()) for a, w in included.items()}
    for name in ["summary.json", "protocol.json"]:
        source = ROOT / "calibration" / name
        archive = source.with_name(source.stem + "_initial_all4.json")
        if not archive.exists():
            shutil.copy2(source, archive)
    summary.update(launchable=True, weights=selected, requested_weights=requested,
                   selection_amendment=dict(post_calibration=True,
                       reason="The initial all-four gate failed. Apply the existing Qwen A5 axis inclusion rule with unchanged per-axis gates.",
                       dropped=[a for a in requested if a not in included],
                       original_rule=original["arm_A5"]["rule"],
                       original_protocol_sha256=hashlib.sha256(original_protocol.read_bytes()).hexdigest()))
    write(path, summary)
    protocol_path = ROOT / "calibration/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["selection_amendment"] = summary["selection_amendment"]
    write(protocol_path, protocol)
    judge.update(requested_weights=requested, weights=selected,
                 calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(cfg_path, cfg)
    print(json.dumps(dict(selected_weights=selected, dropped=summary["selection_amendment"]["dropped"]), indent=2))


if __name__ == "__main__":
    main()
