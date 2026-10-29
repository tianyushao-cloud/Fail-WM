"""Offline checks for proxy outages and checkpoint-preserving recovery."""
import ast
import datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import runtime


OK = dict(ok=True, status=200, reason="HTTP_200")
DOWN = dict(ok=False, status=None, reason="URLError")


class NetworkRecoveryTests(unittest.TestCase):
    def test_route_must_recover_and_stay_up(self):
        with patch.object(runtime, "api_health", side_effect=[DOWN, OK, DOWN, OK, OK]) as health, \
                patch.object(runtime.time, "sleep"):
            waits = []
            self.assertEqual(runtime.wait_for_api(on_wait=waits.append), OK)
        self.assertEqual(health.call_count, 5)
        self.assertEqual(len(waits), 2)

    def test_rejected_credentials_do_not_retry(self):
        with patch.object(runtime, "api_health", return_value=dict(ok=False, status=401, reason="HTTP_401")), \
                patch.object(runtime.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                runtime.wait_for_api()
            sleep.assert_not_called()

    def test_outage_wait_is_bounded(self):
        with patch.object(runtime, "api_health", return_value=DOWN), \
                patch.object(runtime.time, "monotonic", side_effect=[0, 0, 1, 1, 3]), \
                patch.object(runtime.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "checkpoints.*preserved"):
                runtime.wait_for_api(max_wait_seconds=2)

    def test_health_diagnostics_never_include_private_urls(self):
        secret = "test-private-value"
        with patch.dict(runtime.os.environ, OPENAI_API_KEY=secret, HTTPS_PROXY="http://user:" + secret + "@proxy:80"), \
                patch.object(runtime.urllib.request, "build_opener") as build:
            build.return_value.open.side_effect = urllib.error.URLError(secret)
            result = runtime.api_health()
        self.assertNotIn(secret, json.dumps(result))
        self.assertEqual(result, DOWN)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, "https://api.openai.com/v1/models")

    def exercise_launcher(self, failure, old_log=""):
        tree = ast.parse(Path(__file__).with_name("run.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        train = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "train")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "train"
            (out / "checkpoints/step_000001").mkdir(parents=True)
            (out / "checkpoints/step_000001/experiment.json").write_text("{}")
            (root / "logs").mkdir()
            log = root / "logs/train.log"
            log.write_text(old_log)
            resumes = []
            recoveries = []

            def write(path, value):
                Path(path).write_text(json.dumps(value))

            def command(stage, params, distributed=False):
                launch = json.loads((root / "train_launch.json").read_text())
                resumes.append(Path(launch["resume"]).name)
                if len(resumes) == 1:
                    (out / "checkpoints/step_000002").mkdir()
                    (out / "checkpoints/step_000002/experiment.json").write_text("{}")
                    with log.open("a") as stream:
                        stream.write(failure)
                    raise subprocess.CalledProcessError(1, ["mock_train"])

            namespace = dict(ROOT=root, args=SimpleNamespace(gpus=4), Path=Path,
                             os=os, datetime=datetime,
                             time=SimpleNamespace(sleep=lambda _: None),
                             write=write, command=command, subprocess=subprocess,
                             network_ready=lambda: recoveries.append(True))
            exec(compile(ast.Module(body=[train], type_ignores=[]), "run.py", "exec"), namespace)
            error = None
            try:
                namespace["train"]("train", {"output": str(out)})
            except subprocess.CalledProcessError as caught:
                error = caught
            return resumes, recoveries, error

    def test_network_retry_resumes_newest_committed_step(self):
        resumes, recoveries, error = self.exercise_launcher("Official Decisions transport failed")
        self.assertEqual(resumes, ["step_000001", "step_000002"])
        self.assertEqual(recoveries, [True])
        self.assertIsNone(error)

    def test_model_failure_is_not_retried_due_to_old_network_error(self):
        resumes, recoveries, error = self.exercise_launcher(
            "CUDA out of memory", old_log="Official Decisions transport failed\n")
        self.assertEqual(resumes, ["step_000001"])
        self.assertEqual(recoveries, [])
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
