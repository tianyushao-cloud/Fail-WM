"""Local Decisions review packets and strict score replay.

The packet/receipt format is OUR offline contract, not an OpenAI API schema.
API mode sends images only to the official Decisions endpoint. Initial observations and predictions are
exported; future ground truth stays inside the existing reconstruction reward.
"""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .batch import TensorBatch
from .rewards import CodecWorker


QUESTIONS = [
    {
        "type": "score", "name": name, "instructions": instruction,
        "levels": [
            {"label": "incorrect", "description": "Clear visible violation of this criterion."},
            {"label": "minor_error", "description": "A minor visible violation; the main state remains coherent."},
            {"label": "consistent", "description": "No visible violation in the supplied sequence."},
        ],
    }
    for name, instruction in (
        ("object_consistency", "Track object identity, count and shape through time. Allow ordinary occlusion and leaving the frame; do not confuse these with disappearance."),
        ("temporal_interaction", "Check temporal continuity and visible physical interactions. Penalize teleportation and unsupported motion or contact. Judge the predicted outcome, not task success; a coherent failure can be correct."),
    )
]


class MissingDecisionScores(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    data = (canonical(value) + "\n").encode()
    atomic_write(path, lambda stream: stream.write(data))


def validate_decision_config(cfg, method, world=1):
    reward = cfg.get("reward")
    if method == "sft":
        return
    if reward not in ("R0", "R1", "decisions", "hybrid"):
        raise ValueError(f"Unknown reward: {reward}")
    if reward not in ("decisions", "hybrid"):
        return
    if cfg.get("paper_locked", False):
        raise ValueError("Decisions experiments must set paper_locked=false")
    if world != 1:
        raise ValueError("Offline Decisions collection/replay currently requires one process")
    dc = cfg.get("decisions", {})
    if dc.get("mode", "score") not in ("score", "pairwise"):
        raise ValueError("Unknown Decisions mode")
    if dc.get("rubric", "original") not in ("original", "fine_v2", "reference_v3", "reference_score_v4", "reference_alignment", "focused_alignment", "frame_decisions", "reliable_pair"):
        raise ValueError("Unknown Decisions rubric")
    if dc.get("rubric") == "reference_score_v4":
        value=dc.get("reference_min_range", .02)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Invalid reference_min_range")
    if dc.get("mode") == "pairwise":
        if reward != "decisions":
            raise ValueError("Pairwise mode requires pure decisions reward")
        margin = dc.get("pairwise_min_margin", .05)
        if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 < margin <= 1:
            raise ValueError("Invalid pairwise_min_margin")
        spacing = dc.get("spacing_seconds", 1)
        if type(spacing) not in (int, float) or not math.isfinite(spacing) or spacing < 0:
            raise ValueError("Invalid spacing_seconds")
    if dc.get("backend") not in ("replay", "api"):
        raise ValueError("Decisions backend must be replay or api")
    if dc.get("backend") == "api":
        if dc.get("model") != "gpt-6-luna":
            raise ValueError("Official Decisions currently supports gpt-6-luna")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for Decisions API mode")
    for name, default in (("max_api_calls", 4), ("timeout_seconds", 60), ("retries", 1)):
        value = dc.get(name, default)
        if type(value) is not int or value < (0 if name == "retries" else 1):
            raise ValueError(f"decisions.{name} must be a valid integer")
    for name in ("model", "review_dir"):
        if not isinstance(dc.get(name), str) or not dc[name].strip():
            raise ValueError(f"decisions.{name} must be a nonempty string")
    if dc.get("base_reward", "R0") not in ("R0", "R1"):
        raise ValueError("decisions.base_reward must be R0 or R1")
    for name, default in (("base_scale", 1.0), ("weight", 0.25)):
        value = dc.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"decisions.{name} must be finite and positive")


class DecisionStore:
    def __init__(self, config):
        self.root = Path(config["review_dir"])
        self.model = config["model"]
        self.questions = QUESTIONS
        self.backend = config.get("backend", "replay")
        self.client = None
        if self.backend == "api":
            from .decision_api import DecisionsClient
            self.client = DecisionsClient(config.get("max_api_calls", 4),
                                          config.get("timeout_seconds", 60),
                                          config.get("retries", 1),
                                          spacing=config.get("spacing_seconds", 0))

    def prepare(self, initial, prediction, raw_actions, task, dataset):
        arrays = {
            "initial_rgb": np.ascontiguousarray(initial, dtype=np.uint8),
            "prediction_rgb": np.ascontiguousarray(prediction, dtype=np.uint8),
            "raw_actions": np.ascontiguousarray(raw_actions, dtype="<f4"),
        }
        if not np.isfinite(arrays["raw_actions"]).all():
            raise ValueError("Nonfinite raw actions")
        metadata = dict(
            schema_version=1, scoring_protocol="openai_decisions_images_v1", model=self.model, questions=QUESTIONS,
            task=str(task), dataset=dataset, preprocessing="rgb_uint8_round_v1_all_frames",
            prediction_frame_indices=list(range(2, 2 + len(prediction))),
            initial_frame_indices=[0, 1],
            action_note="Raw dataset actions aligned with prediction indices. Coordinate/physical semantics are not verified; action-consistency scoring is disabled.",
            arrays={k: dict(shape=list(v.shape), dtype=v.dtype.str,
                            sha256=hashlib.sha256(v.tobytes()).hexdigest())
                    for k, v in arrays.items()},
        )
        key = hashlib.sha256(canonical(metadata).encode()).hexdigest()
        folder = self.root / key
        folder.mkdir(parents=True, exist_ok=True)
        packet = dict(request_id=key, **metadata)
        if not (folder / "frames.npz").exists():
            atomic_write(folder / "frames.npz", lambda f: np.savez_compressed(f, **arrays))
        write_json(folder / "request.json", packet)
        return key

    def read(self, key):
        path = self.root / key / "score.json"
        if not path.exists():
            return None
        receipt = json.loads(path.read_text())
        if receipt.get("schema_version") != 1 or receipt.get("request_id") != key or receipt.get("model") != self.model:
            raise ValueError(f"Mismatched score receipt for {key}")
        if not isinstance(receipt.get("provenance"), str) or not receipt["provenance"].strip():
            raise ValueError("Score receipt must record nonempty provenance")
        if "official_response" in receipt:
            from .decision_api import normalized_scores, PROTOCOL
            if receipt.get("scoring_protocol") != PROTOCOL:
                raise ValueError("Unknown Decisions scoring protocol")
            values = list(normalized_scores(receipt["official_response"], self.questions, self.model).values())
        else:
            if self.backend == "api":
                raise ValueError("API mode requires an official response receipt")
            labels = receipt.get("labels")
            expected = {q["name"] for q in self.questions}
            if not isinstance(labels, dict) or set(labels) != expected:
                raise ValueError("Score labels must match all rubric names exactly")
            values = []
            for q in self.questions:
                levels = [v["label"] for v in q["levels"]]
                if labels[q["name"]] not in levels:
                    raise ValueError(f"Unknown level for {q['name']}")
                values.append(levels.index(labels[q["name"]]) / (len(levels) - 1))
        # Once a receipt has driven training it cannot silently change on resume.
        accepted = path.with_name("accepted_score.json")
        if accepted.exists() and json.loads(accepted.read_text()) != receipt:
            raise ValueError("Previously consumed score receipt changed; use a fresh review directory/run")
        if not accepted.exists():
            write_json(accepted, receipt)
        return sum(values) / len(values)


    @staticmethod
    def build_payload(packet, arrays):
        from .decision_api import build_request
        return build_request(packet, arrays)

    def request_score(self, key):
        from .decision_api import build_request, PROTOCOL
        if self.client is None:
            raise ValueError("API client is not enabled")
        folder = self.root / key
        packet = json.loads((folder / "request.json").read_text())
        identity = {k: v for k, v in packet.items() if k != "request_id"}
        if packet["request_id"] != key or hashlib.sha256(canonical(identity).encode()).hexdigest() != key:
            raise ValueError("Review metadata integrity mismatch")
        with np.load(folder / "frames.npz", allow_pickle=False) as arrays:
            for name, expected in packet["arrays"].items():
                array = arrays[name]
                if (list(array.shape) != expected["shape"] or array.dtype.str != expected["dtype"]
                    or hashlib.sha256(array.tobytes()).hexdigest() != expected["sha256"]):
                    raise ValueError("Review array integrity mismatch")
            payload = self.build_payload(packet, arrays)
        response, transport = self.client.score(payload)
        write_json(folder / "score.json", dict(
            schema_version=1, request_id=key, model=self.model, scoring_protocol=PROTOCOL,
            provenance="OpenAI official Decisions API", transport=transport,
            official_response=response,
        ))


class DecisionReward:
    def __init__(self, codec, config, dataset):
        self.worker = CodecWorker(codec, None)
        self.store = DecisionStore(config)
        self.dataset = dataset

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels, *, task, raw_actions):
        codes = torch.stack([f["response"] for f in frames], 1)
        if codes.min() < 0 or codes.max() >= 4375:
            raise ValueError("Invalid visual tokens")
        pred = self.worker.detokenize(TensorBatch(dict(
            tokens=codes, ctx_tokens=context.expand(len(codes), -1, -1)
        )), None).batch["pixels"][:, 1:]
        if pred.shape[1] != len(frames) or pixels.shape[1] < 2:
            raise ValueError("Unexpected prediction/initial observation length")
        def rgb(t):
            if not torch.isfinite(t).all():
                raise ValueError("Nonfinite RGB frames")
            return t.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        # Never pass pixels[:, 2:] to the reviewer.
        initial = rgb(pixels[0, :2])
        raw = np.asarray(raw_actions, dtype=np.float32)
        if raw.ndim != 2 or len(raw) < 2 + len(frames):
            raise ValueError("Expected raw actions including two context steps")
        keys = [self.store.prepare(initial, rgb(p), raw[2:2 + len(frames)], task, self.dataset) for p in pred]
        scores = []
        for key in keys:
            score = self.store.read(key)
            if score is None and self.store.client is not None:
                self.store.request_score(key)
                score = self.store.read(key)
            scores.append(score)
        missing = [key for key, score in zip(keys, scores) if score is None]
        if missing:
            raise MissingDecisionScores(
                f"{len(missing)} candidate scores missing; all review packets exported to {self.store.root}. No fabricated scores or optimizer update. Complete score.json receipts, then restart from the same checkpoint/config."
            )
        rewards = torch.tensor(scores, dtype=torch.float32, device=codes.device)
        return rewards, {
            "decisions/api_calls": float(self.store.client.calls if self.store.client else 0),
            "decisions/mean": float(rewards.mean()),
            "decisions/std": float(rewards.std(unbiased=False)),
            "decisions/all_tied": float(torch.all(rewards == rewards[0])),
        }
