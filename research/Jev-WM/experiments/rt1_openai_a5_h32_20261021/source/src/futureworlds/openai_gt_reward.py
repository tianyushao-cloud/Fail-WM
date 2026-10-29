"""Same Qwen axis aggregation and MemSPO reward hook, backed by Decisions."""
import hashlib
import json
from pathlib import Path

from .openai_judge import OpenAIDecisionsJudge, PROTOCOL
from .qwen_gt_reward import QwenGTReward
from .rewards import CodecWorker


class OpenAIGTReward(QwenGTReward):
    def __init__(self, codec, config, device, horizon):
        path = Path(config["calibration_file"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != config["calibration_sha256"]:
            raise ValueError("OpenAI calibration checksum mismatch")
        calibration = json.loads(data)
        if not calibration["launchable"] or calibration["protocol"] != PROTOCOL:
            raise ValueError("Decisions calibration has not passed")
        if calibration["model"] != config["model"] or calibration["scale"] != config["scale"]:
            raise ValueError("Decisions judge differs from calibration")
        if calibration["weights"] != config["weights"]:
            raise ValueError("Reward axes/weights differ from calibration")
        self.worker = CodecWorker(codec, None)
        self.judge = OpenAIDecisionsJudge(config)
        self.weights = {k: float(v) for k, v in config["weights"].items()}
        self.taus = {k: float(calibration["axes"][k]["tau"]) for k in self.weights}
        self.horizon = horizon
        self.root = Path(config["log_dir"])
