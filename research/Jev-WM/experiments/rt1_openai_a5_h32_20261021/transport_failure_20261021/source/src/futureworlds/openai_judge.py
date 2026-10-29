"""Decisions score adapter for the unchanged Qwen multi-axis evidence builders."""
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from .decision_api import DecisionsClient, ENDPOINT, normalized_scores
from .jev_judge import atomic_json, canonical
from .qwen_judge import QwenJudge, SUFFIX

PROTOCOL = "openai_pointwise_qwen_axes_v1"


def score_question(text, name):
    text = text.removesuffix("\n" + SUFFIX).removesuffix(SUFFIX).strip()
    pattern = r"(?:^|[;\n])\s*([0-4])\s*=\s*(.*?)(?=(?:[;\n]\s*[0-4]\s*=)|$)"
    levels = {int(k): v.strip(" ;\n") for k, v in re.findall(pattern, text, re.S)}
    if set(levels) != set(range(5)):
        raise ValueError("Expected the original five explicit rubric levels")
    return dict(type="score", name=name, instructions=text,
                levels=[dict(label=str(k), description=levels[k]) for k in range(5)])


class OpenAIDecisionsJudge(QwenJudge):
    # QwenJudge's image/message builders remain available; no local judge loads.
    def __init__(self, config):
        self.config = dict(config)
        self.scale = int(config["scale"])
        self.root = Path(config["review_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_name = config["model"]
        if self.model_name != "gpt-6-luna":
            raise ValueError("Use the supported Decisions model gpt-6-luna")
        self.workers = int(config.get("workers", 4))
        self._image_cache = {}

    def evidence(self, messages):
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Expected one user evidence message")
        content = messages[0]["content"]
        if content[-1]["type"] != "text":
            raise ValueError("Missing original rubric question")
        manifest = []
        for item in content[:-1]:
            if item["type"] == "text":
                manifest.append(dict(type="input_text", text=item["text"]))
            elif item["type"] == "image":
                image = item["image"]
                raw_key = hashlib.sha256(str((image.mode, image.size)).encode() + image.tobytes()).hexdigest()
                digest = self._image_cache.get(raw_key)
                if digest is None:
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    data = stream.getvalue()
                    digest = hashlib.sha256(data).hexdigest()
                    path = self.root / "images" / (digest + ".png")
                    path.parent.mkdir(exist_ok=True)
                    if not path.exists():
                        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".image-")
                        try:
                            with os.fdopen(fd, "wb") as f:
                                f.write(data)
                            os.replace(tmp, path)
                        finally:
                            if os.path.exists(tmp):
                                os.unlink(tmp)
                    self._image_cache[raw_key] = digest
                manifest.append(dict(type="input_image", sha256=digest))
            else:
                raise ValueError("Unexpected evidence type")
        return manifest, content[-1]["text"]

    def reserve_attempt(self):
        with (self.root / ".budget.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / "http_budget.json"
            state = json.loads(path.read_text()) if path.exists() else dict(attempts=0, next_slot=0.)
            if state["attempts"] >= self.config["max_http_attempts"]:
                raise RuntimeError("Shared Decisions HTTP attempt budget exhausted")
            slot = max(time.time(), state["next_slot"])
            state.update(attempts=state["attempts"] + 1,
                         next_slot=slot + self.config.get("spacing_seconds", .25))
            atomic_json(path, state)
        time.sleep(max(0, slot - time.time()))

    def request(self, identity):
        key = hashlib.sha256(canonical(identity).encode()).hexdigest()
        folder = self.root / "requests" / key
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            atomic_json(folder / "request.json", identity)
            receipt_path = folder / "receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["request_sha256"] != key:
                    raise ValueError("Cached receipt identity mismatch")
            else:
                content = []
                for item in identity["input_manifest"]:
                    if item["type"] == "input_text":
                        content.append(item)
                    else:
                        data = (self.root / "images" / (item["sha256"] + ".png")).read_bytes()
                        if hashlib.sha256(data).hexdigest() != item["sha256"]:
                            raise ValueError("Evidence image checksum mismatch")
                        content.append(dict(type="input_image", image_url="data:image/png;base64," + base64.b64encode(data).decode("ascii")))
                payload = dict(model=self.model_name, input=[dict(role="user", content=content)], questions=identity["questions"])
                for attempt in range(self.config.get("retries", 1) + 1):
                    self.reserve_attempt()
                    client = DecisionsClient(max_calls=1, timeout=self.config.get("timeout_seconds", 60), retries=0)
                    try:
                        response, transport = client.score(payload)
                        break
                    except Exception as error:
                        atomic_json(folder / "last_error.json", dict(error=str(error), http=client.last_error, unix=time.time()))
                        status = (client.last_error or {}).get("status")
                        if status not in (429, 500, 502, 503, 504) or attempt == self.config.get("retries", 1):
                            raise
                        retry_after = (client.last_error or {}).get("headers", {}).get("retry-after", "0")
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = 0
                        if wait > 60:
                            raise
                        time.sleep(max(2, wait))
                receipt = dict(request_sha256=key, response=response, transport=transport,
                               provenance="OpenAI Decisions API", created_unix=time.time())
                atomic_json(receipt_path, receipt)
            normalized_scores(receipt["response"], identity["questions"], self.model_name)
            digest = hashlib.sha256(canonical(receipt).encode()).hexdigest()
            accepted = folder / "accepted_sha256.json"
            if accepted.exists() and json.loads(accepted.read_text())["sha256"] != digest:
                raise ValueError("Previously consumed Decisions receipt changed")
            atomic_json(accepted, dict(sha256=digest))
            return receipt["response"]

    def level_logits(self, items):
        # Identical evidence (e.g. fidelity's structure/dynamics/interaction)
        # shares one HTTP request with separate typed score questions.
        groups = {}
        for i, messages in enumerate(items):
            manifest, text = self.evidence(messages)
            key = hashlib.sha256(canonical(manifest).encode()).hexdigest()
            entry = groups.setdefault(key, dict(manifest=manifest, questions=[], indices=[], question_map={}))
            question = score_question(text, "")
            question_key = canonical(question)
            if question_key not in entry["question_map"]:
                question["name"] = f"q{len(entry['questions'])}"
                entry["questions"].append(question)
                entry["question_map"][question_key] = question["name"]
            entry["indices"].append((i, entry["question_map"][question_key]))
        out = np.empty((len(items), 5), dtype=np.float64)

        def evaluate(entry):
            identity = dict(protocol=PROTOCOL, endpoint=ENDPOINT, model=self.model_name,
                            input_manifest=entry["manifest"], questions=entry["questions"])
            response = self.request(identity)
            by_name = {a["name"]: a for a in response["answers"]}
            for i, name in entry["indices"]:
                values = sorted(by_name[name]["probabilities"], key=lambda p: p["value"])
                p = np.array([x["probability"] for x in values], dtype=np.float64)
                out[i] = np.log(p, where=p > 0, out=np.full(5, -np.inf))

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(evaluate, groups.values()))
        return out
