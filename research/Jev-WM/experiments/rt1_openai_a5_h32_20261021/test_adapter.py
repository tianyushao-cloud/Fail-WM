"""Offline contract tests; no API calls or CUDA allocation."""
import json
import http.client
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source/src"))
from futureworlds.openai_judge import OpenAIDecisionsJudge, score_question
from futureworlds.qwen_axes import PHYSICS_Q, TASK_Q, SMOOTH_Q, fidelity, motion, reference_free
from futureworlds.qwen_judge import QUESTIONS, STALL_QUESTION, SUFFIX


class FakeJudge(OpenAIDecisionsJudge):
    def request(self, identity):
        self.identities.append(identity)
        probabilities = [.1, .1, .2, .2, .4]
        return dict(model=self.model_name, answers=[dict(type="score", name=q["name"],
                    probabilities=[dict(value=i, label=str(i), probability=p) for i, p in enumerate(probabilities)])
                    for q in identity["questions"]])


class AdapterTests(unittest.TestCase):
    def test_disconnect_retries_then_replays_saved_receipt(self):
        calls = []

        class Client:
            last_error = None

            def __init__(self, **kwargs):
                pass

            def score(self, payload):
                calls.append(payload)
                if len(calls) == 1:
                    raise http.client.RemoteDisconnected("Remote end closed connection")
                q = payload["questions"][0]
                p = [.1, .1, .2, .2, .4]
                return dict(model=payload["model"], answers=[dict(type="score", name=q["name"],
                    score=2.7, confidence=.4, probabilities=[dict(value=i, label=str(i), probability=v)
                    for i, v in enumerate(p)])]), dict(request_id="offline-test")

        with tempfile.TemporaryDirectory() as root, patch("futureworlds.openai_judge.DecisionsClient", Client), patch("futureworlds.openai_judge.time.sleep"):
            judge = OpenAIDecisionsJudge(dict(scale=1, review_dir=root, model="gpt-6-luna", retries=1,
                                              max_http_attempts=3, spacing_seconds=0))
            identity = dict(model="gpt-6-luna", input_manifest=[], questions=[score_question(QUESTIONS["structure"], "q0")])
            first = judge.request(identity)
            self.assertEqual(judge.request(identity), first)
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads((Path(root) / "http_budget.json").read_text())["attempts"], 2)

    def test_invalid_score_is_not_retried_or_cached(self):
        calls = []

        class Client:
            last_error = None

            def __init__(self, **kwargs):
                pass

            def score(self, payload):
                calls.append(payload)
                raise ValueError("Invalid Decisions probabilities")

        with tempfile.TemporaryDirectory() as root, patch("futureworlds.openai_judge.DecisionsClient", Client):
            judge = OpenAIDecisionsJudge(dict(scale=1, review_dir=root, model="gpt-6-luna", retries=2,
                                              max_http_attempts=3, spacing_seconds=0))
            identity = dict(model="gpt-6-luna", input_manifest=[], questions=[score_question(QUESTIONS["structure"], "q0")])
            with self.assertRaisesRegex(ValueError, "Invalid Decisions"):
                judge.request(identity)
            self.assertEqual(len(calls), 1)
            self.assertFalse(list(Path(root).glob("requests/*/receipt.json")))

    def test_original_rubrics_preserve_all_levels(self):
        for text in list(QUESTIONS.values()) + [PHYSICS_Q, TASK_Q, SMOOTH_Q, STALL_QUESTION]:
            q = score_question(text + "\n" + SUFFIX if not text.endswith(SUFFIX) else text, "test")
            self.assertEqual([x["label"] for x in q["levels"]], list("01234"))
            self.assertTrue(all(x["description"] for x in q["levels"]))
            self.assertNotIn(SUFFIX, q["instructions"])

    def test_invalid_rubric_fails(self):
        with self.assertRaises(ValueError):
            score_question("Rate 0 to 4", "bad")

    def test_evidence_grouping_and_probability_expectation(self):
        with tempfile.TemporaryDirectory() as root:
            judge = FakeJudge(dict(scale=1, review_dir=root, model="gpt-6-luna", workers=1))
            judge.identities = []
            frame = np.zeros((32, 32, 3), np.uint8)
            messages = [[dict(role="user", content=[dict(type="text", text="context"),
                        dict(type="image", image=Image.fromarray(frame)), dict(type="text", text=q + "\n" + SUFFIX)])]
                        for q in QUESTIONS.values()]
            logits = judge.level_logits(messages + [messages[0]])
            np.testing.assert_allclose(np.exp(logits) @ np.arange(5) / 4, [.675] * 4)
            np.testing.assert_array_equal(logits[0], logits[3])
            self.assertEqual(len(judge.identities), 1)
            self.assertEqual(len(judge.identities[0]["questions"]), 3)
            frame[0, 0] = 255
            changed = [[dict(role="user", content=[dict(type="text", text="context"),
                       dict(type="image", image=Image.fromarray(frame)), dict(type="text", text=QUESTIONS["structure"] + "\n" + SUFFIX)])]]
            judge.level_logits(changed)
            self.assertNotEqual(judge.identities[0]["input_manifest"], judge.identities[1]["input_manifest"])

    def test_original_axis_builders_accept_adapter_without_local_model(self):
        with tempfile.TemporaryDirectory() as root:
            judge = FakeJudge(dict(scale=1, review_dir=root, model="gpt-6-luna", workers=1))
            judge.identities = []
            initial = np.zeros((2, 32, 32, 3), np.uint8)
            clip = np.zeros((32, 32, 32, 3), np.uint8)
            predictions = np.stack([clip] * 4)
            for fn in [lambda: fidelity(judge, initial, clip, predictions, "pick object"),
                       lambda: motion(judge, initial, clip, predictions, "pick object"),
                       lambda: reference_free(judge, initial, predictions, "pick object", "physics"),
                       lambda: reference_free(judge, initial, predictions, "pick object", "task")]:
                result = fn()
                np.testing.assert_allclose(result["score"], [.675] * 4)
                self.assertEqual(result["views"].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
