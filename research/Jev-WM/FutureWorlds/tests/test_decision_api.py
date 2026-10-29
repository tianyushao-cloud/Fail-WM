"""Validate official wire adapter offline, including fractional rewards."""
import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
from futureworlds.decision_api import build_request, normalized_scores, DecisionsClient, PROTOCOL
from futureworlds.decision_reward import DecisionStore, QUESTIONS, write_json


def response():
    return dict(model="gpt-6-luna", answers=[dict(
        type="score", name=q["name"], score=1.1, confidence=0.55,
        probabilities=[dict(value=i, label=level["label"], probability=p)
                       for i, (level, p) in enumerate(zip(q["levels"], [0.1, 0.7, 0.2]))]
    ) for q in QUESTIONS], usage={"input_tokens": 100})


class DecisionAPIChecks(unittest.TestCase):
    def test_probability_weighted_score_and_invalid_responses(self):
        good = response()
        self.assertEqual(list(normalized_scores(good, QUESTIONS, "gpt-6-luna").values()), [0.55, 0.55])
        for field, value in [("score", float("nan")), ("score", 3), ("score", True),
                             ("type", "refusal"), ("confidence", -1), ("probabilities", [])]:
            bad = copy.deepcopy(good); bad["answers"][0][field] = value
            with self.assertRaises(ValueError): normalized_scores(bad, QUESTIONS, "gpt-6-luna")
        bad = copy.deepcopy(good); bad["answers"][1]["name"] = QUESTIONS[0]["name"]
        with self.assertRaises(ValueError): normalized_scores(bad, QUESTIONS, "gpt-6-luna")

    def test_image_payload_receipt_cache_and_tamper(self):
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}):
            store = DecisionStore(dict(review_dir=d, model="gpt-6-luna", backend="api"))
            key = store.prepare(np.zeros((2, 8, 8, 3),dtype=np.uint8),
                                np.ones((4, 8, 8, 3),dtype=np.uint8),
                                np.zeros((4, 13)), "move", "robocasa")
            folder = Path(d) / key
            packet = json.loads((folder / "request.json").read_text())
            with np.load(folder / "frames.npz") as arrays:
                payload = build_request(packet, arrays)
            parts = payload["input"][0]["content"]
            self.assertEqual(len([p for p in parts if p["type"] == "input_image"]), 6)
            self.assertTrue(all(p["image_url"].startswith("data:image/png;base64,") for p in parts if p["type"] == "input_image"))
            with patch.object(store.client, "score", return_value=(response(), {"request_id":"test-only"})) as call:
                store.request_score(key)
                call.assert_called_once()
            self.assertAlmostEqual(store.read(key), 0.55)
            packet["task"] = "tampered"; write_json(folder / "request.json", packet)
            with self.assertRaisesRegex(ValueError, "integrity"):
                store.request_score(key)

    def test_http_failures_retry_budget_and_key_not_leaked(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}):
            client = DecisionsClient(max_calls=1, retries=1)
            payload = {"model":"gpt-6-luna", "questions":QUESTIONS, "input":"test"}
            error = urllib.error.HTTPError("https://api.openai.com/v1/decisions", 429, "limited", {}, io.BytesIO(b"test-only"))
            with patch.object(client.opener, "open", side_effect=error), patch("time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "budget"):
                    client.score(payload)
            self.assertEqual(client.calls, 1)
            client = DecisionsClient(max_calls=2)
            error = urllib.error.HTTPError("https://api.openai.com/v1/decisions", 401, "unauthorized", {}, io.BytesIO(b"test-only"))
            with patch.object(client.opener, "open", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                    client.score(payload)
            self.assertNotIn("test-only",str(caught.exception))
            self.assertEqual(client.calls, 1)


if __name__ == "__main__": unittest.main()
