"""Package batch1000_failure70 val/test episodes that are NOT in the original val/test manifests.

failure70 reuses most batch1000 episodes (same episode_id); only distinct ids are new.
Same conversion as the SFT data (prepare_robocasa_sft.convert_episode), same initial states as
the original val/test, so these extend the sibling test without crossing the split.
Output: <out>/{val,test}_extra.json + cases_{split}/NNNNNN.npz
"""
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/media/tyshao/fail_WM/FutureWorlds_SFT")
from prepare_robocasa_sft import convert_episode, sha256, write_manifest

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data")
SRC = Path("/media/tyshao/fail_WM/RoboCasa/data/batch1000_failure70")
old = {}
for n in ("train_natural", "val", "test", "train_scholar_failure_augmented"):
    for r in json.loads((DATA / f"{n}.json").read_text())["cases"]:
        old[r["id"]] = r
rows = [json.loads(l) for l in (SRC / "manifest.jsonl").read_text().splitlines() if l]
for split in ("val", "test"):
    groups = {r["initial_state_id"] for r in json.loads((DATA / f"{split}.json").read_text())["cases"]}
    new = sorted((r for r in rows if r["split"] == split and r["episode_id"] not in old), key=lambda r: r["episode_id"])
    assert all(r["initial_state_id"] in groups for r in new), "extra episode outside the split's initial states"
    cases_dir = OUT / f"cases_{split}"; cases_dir.mkdir(exist_ok=True)
    cases = [convert_episode(r, SRC, cases_dir, "Pick up the cube and place it at the goal.", i) for i, r in enumerate(new)]
    assert len({c["id"] for c in cases}) == len(cases)
    write_manifest(OUT / f"{split}_extra.json", cases, sha256(SRC / "manifest.jsonl"), f"{split}_failure70_extra")
    print(split, len(cases), dict(Counter(c["failure_type"] for c in cases)),
          "frames min", min(c["frames"] for c in cases), flush=True)
