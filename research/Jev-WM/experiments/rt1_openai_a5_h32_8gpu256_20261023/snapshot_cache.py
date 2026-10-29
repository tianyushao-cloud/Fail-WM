"""Snapshot immutable scoring receipts; use independent locks and HTTP counter."""
import errno
import fcntl
import os
from pathlib import Path
import shutil
import time

from continuation import ROOT, OLD, read, write


def copy_once(source, target):
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        return False
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)
    return True


def snapshot_cache():
    source, target = OLD / "reviews", ROOT / "reviews"
    target.mkdir(exist_ok=True)
    images = requests = 0
    for path in (source / "images").glob("*.png"):
        images += copy_once(path, target / "images" / path.name)
    for folder in (source / "requests").iterdir():
        if not folder.is_dir() or not (folder / "receipt.json").is_file():
            continue
        for name in ["request.json", "receipt.json", "accepted_sha256.json"]:
            path = folder / name
            if path.is_file():
                copied = copy_once(path, target / "requests" / folder.name / name)
                requests += int(copied and name == "receipt.json")
    # Never share budget lock files or writable counters across containers.
    source_budget = read(source / "http_budget.json")
    with (target / ".budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = target / "http_budget.json"
        budget = read(path) if path.exists() else dict(attempts=0, next_slot=0.)
        budget["attempts"] = max(budget["attempts"], source_budget["attempts"])
        budget["next_slot"] = max(budget["next_slot"], source_budget["next_slot"])
        write(path, budget)
    result = dict(source=str(source), unix=time.time(), added_images=images,
                  added_receipts=requests, retained_http_attempts=budget["attempts"],
                  independent_locks_and_counter=True)
    write(ROOT / "cache_snapshot.json", result)
    print(f"Cache snapshot: +{requests} receipts; retained HTTP attempt counter {budget['attempts']}", flush=True)
    return result


if __name__ == "__main__":
    lock = (ROOT / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot_cache()
