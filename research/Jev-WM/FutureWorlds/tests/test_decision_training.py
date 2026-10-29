"""Actual one-step training tests using a tiny model/codec and test-only labels."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file
import test_pipeline
from test_training_paths import SquaredDistance
from test_decision_reward import receipt
from futureworlds.data import write_json
from futureworlds.decision_reward import DecisionStore, MissingDecisionScores
from futureworlds.training import train


class DecisionTrainingChecks(unittest.TestCase):
    def test_decisions_and_hybrid_update_and_missing_scores_do_not(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helper = test_pipeline.PortablePipelineChecks()
            helper.make_bundle(root / "bundle")
            helper.make_data(root / "data")
            before = load_file(str(root / "bundle/bridge/models/main/model.safetensors"))
            cfg = dict(method="memspo", dataset="bridge", bundle=str(root / "bundle"),
                       train_manifest=str(root / "data/manifest.json"), device="cpu",
                       seed=23, global_batch=1, steps=1, lr=1e-6, horizon=1,
                       reward="decisions", clip=0.2, kl_coef=0.001, checkpoint_every=1,
                       decisions=dict(backend="replay", model="gpt-6-luna",
                                      review_dir=str(root / "reviews")))
            cfg["output"] = str(root / "missing")
            write_json(root / "config.json", cfg)
            with patch("torch.optim.AdamW.step") as optimizer_step:
                with self.assertRaises(MissingDecisionScores):
                    train(root / "config.json")
                optimizer_step.assert_not_called()
            self.assertFalse((root / "missing/complete.json").exists())
            self.assertFalse((root / "missing/checkpoints").exists())
            self.assertGreater(len(list((root / "reviews").glob("*/request.json"))), 0)
            for mode in ("decisions", "hybrid"):
                cfg["reward"] = mode
                cfg["output"] = str(root / mode)
                cfg["decisions"]["review_dir"] = str(root / (mode + "-reviews"))
                write_json(root / "config.json", cfg)
                seen = {}
                original = DecisionStore.prepare
                def labeled(store, *args, **kwargs):
                    key = original(store, *args, **kwargs)
                    if key not in seen:
                        seen[key] = len(seen)
                    label = ["incorrect", "minor_error", "consistent"][seen[key] % 3]
                    write_json(store.root / key / "score.json", receipt(store, key, label))
                    return key
                with patch.object(DecisionStore, "prepare", labeled), patch("lpips.LPIPS", return_value=SquaredDistance()):
                    train(root / "config.json")
                self.assertGreater(len(seen), 1)
                after = load_file(str(root / mode / "checkpoints/step_000001/model.safetensors"))
                self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before), mode)
                self.assertEqual(json.loads((root / mode / "complete.json").read_text())["state"], "COMPLETE")
                metrics = json.loads((root / mode / "metrics.jsonl").read_text())
                self.assertIn("decisions/mean", metrics["reward_metrics"])


if __name__ == "__main__":
    unittest.main()
