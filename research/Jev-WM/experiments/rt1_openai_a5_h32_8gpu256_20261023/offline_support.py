"""Shared offline test fixtures; never writes real experiment scores."""
import json

from pathlib import Path

import subprocess

import tempfile

import unittest

from unittest.mock import patch

import continuation as c

import run

import runtime

def checkpoint(output, step, fp="test"):
    folder = Path(output) / "checkpoints" / f"step_{step:06d}"
    folder.mkdir(parents=True)
    for name in ("model.safetensors", "text_adapters.safetensors", "config.json", "training_state.pt"):
        (folder / name).write_bytes(b"unit-test-only")
    c.write(folder / "experiment.json", dict(step=step, fingerprint=fp,
            files={p.name: c.digest(p) for p in folder.iterdir()}))
    return folder

def metrics(output, steps):
    Path(output, "metrics.jsonl").write_text("".join(json.dumps(dict(step=s)) + "\n" for s in steps))

class CheckpointTests(unittest.TestCase):
    def test_partial_never_selected_and_bad_latest_never_silently_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            good = checkpoint(d, 10)
            bad = checkpoint(d, 99)
            bad.rename(bad.with_name(bad.name + ".partial"))
            self.assertEqual(c.latest_checkpoint(d, "test"), (good, 10))
            bad = checkpoint(d, 11)
            (bad / "training_state.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                c.latest_checkpoint(d, "test")

    def test_recovery_archives_uncommitted_rows_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint(d, 10)
            partial = Path(d) / "checkpoints/step_000011.partial"
            partial.mkdir()
            metrics(d, list(range(1, 12)) + [11])
            c.clean_uncommitted(d, 10)
            self.assertFalse(partial.exists())
            self.assertEqual(len(Path(d, "metrics.jsonl").read_text().splitlines()), 10)
            self.assertEqual(len(list(Path(d, "recovery_archive").iterdir())), 2)

    def test_exact_optimizer_hash_detects_moment_changes(self):
        import torch
        state = dict(state={0: dict(step=torch.tensor(10.), exp_avg=torch.ones(3), exp_avg_sq=torch.ones(3))},
                     param_groups=[dict(lr=1e-6, params=[0])])
        import copy
        other = copy.deepcopy(state)
        self.assertEqual(c.state_digest(state), c.state_digest(other))
        other["state"][0]["exp_avg"][1] += .01
        self.assertNotEqual(c.state_digest(state), c.state_digest(other))

class NetworkTests(unittest.TestCase):
    def test_route_needs_two_consecutive_successes(self):
        ok = dict(ok=True, status=200, reason="HTTP_200")
        down = dict(ok=False, status=None, reason="URLError")
        with patch.object(runtime, "api_health", side_effect=[down, ok, down, ok, ok]), \
                patch.object(runtime.time, "sleep"):
            waits = []
            self.assertEqual(runtime.wait_for_api(on_wait=waits.append), ok)
            self.assertEqual(len(waits), 2)

    def test_auth_rejection_fails_fast(self):
        with patch.object(runtime, "api_health", return_value=dict(ok=False, status=401, reason="HTTP_401")), \
                patch.object(runtime.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                runtime.wait_for_api()
            sleep.assert_not_called()

    def test_wait_is_bounded_and_preserves_state(self):
        with patch.object(runtime, "api_health", return_value=dict(ok=False, status=None, reason="URLError")), \
                patch.object(runtime.time, "monotonic", side_effect=[0, 0, 1, 1, 3]), \
                patch.object(runtime.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "checkpoints.*preserved"):
                runtime.wait_for_api(max_wait_seconds=2)
