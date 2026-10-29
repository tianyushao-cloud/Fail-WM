"""Offline tests: no API calls, real training, or modification to old experiments."""
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


class LauncherTests(unittest.TestCase):
    def test_outage_resumes_committed_step_and_cleans_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as d:
            root, output = Path(d), Path(d) / "train"
            (root / "logs").mkdir()
            checkpoint(output, 10)
            metrics(output, range(1, 11))
            launched = []

            def child(*args, **kwargs):
                launch = c.read(root / "train_launch.json")
                launched.append(launch)
                self.assertEqual(launch["resume"], str(output / "checkpoints/step_000010"))
                if len(launched) == 1:
                    metrics(output, range(1, 12))
                    (output / "checkpoints/step_000011.partial").mkdir()
                    (root / "logs/train.log").write_text("Official Decisions transport failed; no score produced\n")
                    raise subprocess.CalledProcessError(1, ["unit-test-child"])
                self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 10)
                self.assertFalse((output / "checkpoints/step_000011.partial").exists())
                checkpoint(output, 100)
                metrics(output, range(1, 101))
                c.write(output / "complete.json", dict(steps=100, fingerprint="test"))

            with patch.object(run, "ROOT", root), patch.object(run, "command", side_effect=child), \
                    patch.object(run, "network_ready") as ready, patch.object(run.time, "sleep") as sleep:
                run.train(dict(output=str(output), steps=100), "test")
                self.assertEqual(ready.call_count, 1)
                sleep.assert_called_once_with(30)
            self.assertEqual(len(launched), 2)

    def test_old_ten_step_complete_marker_does_not_skip(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint(d, 10)
            metrics(d, range(1, 11))
            c.write(Path(d) / "complete.json", dict(steps=10, fingerprint="test"))
            with patch.object(run, "command") as child:
                with self.assertRaisesRegex(ValueError, "completion marker"):
                    run.train(dict(output=d), "test")
                child.assert_not_called()

    def test_non_network_errors_do_not_retry(self):
        for message in ("CUDA out of memory", "Shared Decisions HTTP attempt budget exhausted",
                        "Official Decisions HTTP 401", "Official Decisions HTTP 403",
                        "Official Decisions HTTP 400", "Checkpoint checksum mismatch",
                        "bad probabilities; Official Decisions HTTP 500"):
            self.assertFalse(run.network_failure(message), message)
        for code in (429, 500, 502, 503, 504):
            self.assertTrue(run.network_failure(f"Official Decisions HTTP {code}"))


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


class ConfigurationTests(unittest.TestCase):
    def test_no_learning_rate_reward_or_schedule_drift(self):
        import copy
        cfg = c.read(c.ROOT / "train.json")
        old = c.read(c.OLD / "train/checkpoints/step_000010/experiment.json")["config"]
        c.validate_change(old, cfg)
        for key, value in (("lr", 2e-6), ("global_batch", 32), ("seed", 42)):
            other = copy.deepcopy(cfg)
            other[key] = value
            with self.assertRaisesRegex(ValueError, "change|requires"):
                c.validate_change(old, other)
        other = copy.deepcopy(cfg)
        other["openai_judge"]["weights"]["task"] += .1
        with self.assertRaisesRegex(ValueError, "scoring protocol"):
            c.validate_change(old, other)

    def test_rank_extension_is_reproducible_and_preserves_existing_streams(self):
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA generator state preparation requires one GPU")
        original = [dict(torch=torch.Generator().manual_seed(i).get_state(),
                         numpy=np.random.RandomState(i).get_state(), python=(i,),
                         generator=torch.Generator(device="cuda:0").manual_seed(i).get_state(),
                         cuda=None) for i in range(4)]
        first, seeds = c.migrated_rng(original, "source-hash")
        second, seeds2 = c.migrated_rng(original, "source-hash")
        self.assertEqual(seeds, seeds2)
        self.assertEqual(c.state_digest(first), c.state_digest(second))
        self.assertEqual(c.state_digest(first[:4]), c.state_digest(original))
        self.assertEqual(len({c.state_digest(s) for s in first}), 8)


class FrozenReferenceTests(unittest.TestCase):
    def test_original_sft_reference_is_independent_of_actor_adapters(self):
        import torch
        from futureworlds.checkpoints import load_world_model
        from futureworlds.training import build_reference
        torch.set_num_threads(1)
        cfg = c.read(c.ROOT / "train.json")
        actor = load_world_model(cfg["bundle"], "rt1", "sft", "cpu", verify=False)
        ref = build_reference(actor, cfg, "cpu")
        torch.manual_seed(123)
        width = next(iter(actor.adapters.values())).text_norm.normalized_shape[0]
        text, mask = torch.randn(1, 5, width), torch.ones(1, 5, dtype=torch.bool)
        ids = torch.arange(12)[None] + 4375
        with torch.no_grad():
            before = ref(input_ids=ids, text_features=text, text_mask=mask).logits.clone()
            for adapter in actor.adapters.values():
                adapter.out.weight.add_(.05)
            actor.set_text(text * 3, mask)
            after = ref(input_ids=ids, text_features=text, text_mask=mask).logits
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        self.assertFalse(any(p.requires_grad for p in ref.parameters()))
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(actor.parameters(), ref.parameters())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
