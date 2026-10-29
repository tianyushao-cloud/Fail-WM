"""Offline tests; synthetic labels below are NEVER production reward fallbacks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from futureworlds.batch import TensorBatch
from futureworlds.decision_reward import (
    QUESTIONS, DecisionReward, DecisionStore, MissingDecisionScores,
    validate_decision_config, write_json,
)
from futureworlds.hybrid_reward import HybridReward


def receipt(store, key, label):
    return dict(schema_version=1, request_id=key, model=store.model,
                provenance="synthetic unit-test label; not an API result",
                labels={q["name"]: label for q in QUESTIONS})


class DecisionRewardChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = dict(backend="replay", review_dir=self.tmp.name, model="gpt-6-luna")

    def packet(self, store, task="move", intensity=0):
        return store.prepare(np.zeros((2, 8, 8, 3), dtype=np.uint8),
                             np.full((2, 8, 8, 3), intensity, dtype=np.uint8),
                             np.zeros((2, 3), dtype=np.float32), task, "bridge")

    def test_content_addressing_rubric_model_and_receipt_integrity(self):
        store = DecisionStore(self.cfg)
        key = self.packet(store)
        self.assertEqual(key, self.packet(store))
        self.assertNotEqual(key, self.packet(store, task="other"))
        self.assertNotEqual(key, self.packet(store, intensity=1))
        self.assertNotEqual(key, self.packet(DecisionStore({**self.cfg, "model": "other"})))
        with patch("futureworlds.decision_reward.QUESTIONS", copy.deepcopy(QUESTIONS)) as qs:
            qs[0]["instructions"] += " revised"
            self.assertNotEqual(key, self.packet(store))
        self.assertIsNone(store.read(key))
        path = store.root / key / "score.json"
        write_json(path, receipt(store, key, "minor_error"))
        self.assertEqual(store.read(key), 0.5)
        write_json(path, receipt(store, key, "consistent"))
        with self.assertRaisesRegex(ValueError, "changed"):
            store.read(key)

    def test_reject_malformed_scores(self):
        store = DecisionStore(self.cfg)
        key = self.packet(store)
        good = receipt(store, key, "consistent")
        for update in ({"request_id": "wrong"}, {"model": "wrong"},
                       {"labels": {q["name"]: "invented" for q in QUESTIONS}},
                       {"labels": {}}, {"provenance": ""}):
            write_json(store.root / key / "score.json", {**good, **update})
            with self.assertRaises(ValueError):
                store.read(key)

    def test_export_all_candidates_no_future_gt_and_replay(self):
        reward = DecisionReward(None, self.cfg, "bridge")
        pred = torch.stack([torch.zeros(3, 3, 8, 8), torch.ones(3, 3, 8, 8)])
        reward.worker.detokenize = lambda *args: TensorBatch({"pixels": pred})
        frames = [{"response": torch.zeros(2, 80, dtype=torch.long)} for _ in range(2)]
        context = torch.zeros(1, 1, 1)
        pixels = torch.zeros(1, 4, 3, 8, 8)
        raw = np.arange(12).reshape(4, 3)
        args = (frames, context, None, pixels)
        with self.assertRaises(MissingDecisionScores):
            reward(*args, task="move", raw_actions=raw)
        paths = list(Path(self.tmp.name).glob("*/request.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            with np.load(path.with_name("frames.npz")) as arrays:
                np.testing.assert_array_equal(arrays["raw_actions"], raw[2:])
                self.assertEqual(set(arrays.files), {"initial_rgb", "prediction_rgb", "raw_actions"})
                label = "consistent" if arrays["prediction_rgb"].max() else "incorrect"
            key = path.parent.name
            write_json(path.with_name("score.json"), receipt(reward.store, key, label))
        result, stats = reward(*args, task="move", raw_actions=raw)
        self.assertEqual(result.tolist(), [0.0, 1.0])
        self.assertFalse(result.requires_grad)
        pixels[:, 2:] = float("nan")  # future GT is neither read nor exported
        again, _ = reward(*args, task="move", raw_actions=raw)
        self.assertTrue(torch.equal(result, again))
        self.assertEqual(len(list(Path(self.tmp.name).glob("*/request.json"))), 2)
        self.assertEqual(stats["decisions/all_tied"], 0.0)

    def test_hybrid_fixed_scale_and_missing_score_failure(self):
        base = lambda *a: (torch.tensor([-4., -2.]), {"base": 1.})
        decision = lambda *a, **kw: (torch.tensor([1., 0.]), {"judge": 1.})
        combined = HybridReward(base, decision, base_scale=2, weight=0.5)
        score, _ = combined(None, None, None, None, task="", raw_actions=[])
        torch.testing.assert_close(score, torch.tensor([-1.5, -1.]))
        def missing(*a, **kw):
            raise MissingDecisionScores("missing")
        with self.assertRaises(MissingDecisionScores):
            HybridReward(base, missing)(None, None, None, None, task="", raw_actions=[])
        for val in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                HybridReward(base, decision, weight=val)

    def test_config_fails_early_for_unsupported_modes(self):
        cfg = dict(reward="hybrid", decisions=self.cfg, paper_locked=False)
        validate_decision_config(cfg, "memspo")
        for override in ({"reward": "typo"}, {"paper_locked": True},
                         {"decisions": {**self.cfg, "backend": "api"}},
                         {"decisions": {**self.cfg, "weight": float("nan")}}):
            with self.assertRaises(ValueError):
                validate_decision_config({**cfg, **override}, "memspo")
        with self.assertRaises(ValueError):
            validate_decision_config(cfg, "memspo", world=2)


if __name__ == "__main__":
    unittest.main()
