"""Offline restart tests; synthetic responses stay in temporary test fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import continuation as c
import run
from offline_support import checkpoint, metrics, CheckpointTests, NetworkTests


class RestartTests(unittest.TestCase):
    def test_only_concurrency_spacing_and_paths_may_change(self):
        old = c.read(c.OLD / "train/checkpoints/step_000014/experiment.json")["config"]
        cfg = c.read(c.ROOT / "train.json")
        c.validate_change(old, cfg)
        for key, value in [("lr", 2e-6), ("seed", 42), ("steps", 200), ("global_batch", 32)]:
            bad = copy.deepcopy(cfg)
            bad[key] = value
            with self.assertRaises(ValueError):
                c.validate_change(old, bad)
        bad = copy.deepcopy(cfg)
        bad["openai_judge"]["weights"]["task"] += .1
        with self.assertRaisesRegex(ValueError, "scoring protocol"):
            c.validate_change(old, bad)

    def test_real_thread_pool_can_run_32_distinct_requests_at_once(self):
        import numpy as np
        from futureworlds.openai_judge import OpenAIDecisionsJudge
        cfg = c.read(c.ROOT / "train.json")["openai_judge"]
        with tempfile.TemporaryDirectory() as d:
            cfg["review_dir"] = d
            judge = OpenAIDecisionsJudge(cfg)
            barrier = threading.Barrier(32, timeout=5)
            lock = threading.Lock()
            counts = dict(active=0, maximum=0)
            rubric = "Fixture rubric\n0 = zero; 1 = one; 2 = two; 3 = three; 4 = four."

            def request(identity):
                with lock:
                    counts["active"] += 1
                    counts["maximum"] = max(counts["maximum"], counts["active"])
                barrier.wait()
                with lock:
                    counts["active"] -= 1
                return dict(answers=[dict(name="q0", probabilities=[dict(value=i, probability=.2) for i in range(5)])])

            with patch.object(judge, "evidence", side_effect=lambda i: ([dict(type="input_text", text=str(i))], rubric)), \
                    patch.object(judge, "request", side_effect=request):
                result = judge.level_logits(list(range(32)))
            self.assertEqual(counts["maximum"], 32)
            np.testing.assert_allclose(result, np.log(.2))
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_network_restart_resumes14_and_removes_uncommitted15(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "train"
            (root / "logs").mkdir()
            checkpoint(out, 14)
            metrics(out, range(1, 15))
            calls = []

            def child(*args, **kwargs):
                cfg = c.read(root / "train_launch.json")
                calls.append(cfg)
                self.assertEqual(cfg["resume"], str(out / "checkpoints/step_000014"))
                if len(calls) == 1:
                    (out / "checkpoints/step_000015.partial").mkdir()
                    metrics(out, range(1, 16))
                    (root / "logs/train.log").write_text("Official Decisions transport failed; no score produced")
                    raise subprocess.CalledProcessError(1, ["fixture-child"])
                self.assertFalse((out / "checkpoints/step_000015.partial").exists())
                self.assertEqual(len((out / "metrics.jsonl").read_text().splitlines()), 14)
                checkpoint(out, 100)
                metrics(out, range(1, 101))
                c.write(out / "complete.json", dict(steps=100, fingerprint="test"))

            with patch.object(run, "ROOT", root), patch.object(run, "command", side_effect=child), \
                    patch.object(run, "network_ready") as network, patch.object(run.time, "sleep"):
                run.train(dict(output=str(out), steps=100), "test")
            self.assertEqual(len(calls), 2)
            self.assertEqual(network.call_count, 1)

    def test_prepared_state_preserves_actor_adam_and_all_rank_rng(self):
        import torch
        cfg = c.read(c.ROOT / "train.json")
        target = Path(cfg["output"]) / "checkpoints/step_000014"
        source = c.OLD / "train/checkpoints/step_000014"
        audit = c.read(c.ROOT / "migration.json")
        c.verified_checkpoint(source, c.SOURCE_FINGERPRINT)
        c.verified_checkpoint(target, audit["target_fingerprint"])
        for name in ["model.safetensors", "text_adapters.safetensors"]:
            self.assertEqual(c.digest(source / name), c.digest(target / name))
        old = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
        new = torch.load(target / "training_state.pt", map_location="cpu", weights_only=False)
        self.assertEqual(new["step"], 14)
        self.assertEqual(new["world_size"], 8)
        self.assertEqual(len(new["rng"]), 8)
        self.assertEqual(c.state_digest(old["optimizer"]), c.state_digest(new["optimizer"]))
        self.assertEqual(c.state_digest(old["rng"]), c.state_digest(new["rng"]))
        self.assertEqual(audit["next_step"], 15)
        self.assertEqual(audit["next_schedule_cursor"], 224)
        self.assertFalse((target.parent.parent / "complete.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
