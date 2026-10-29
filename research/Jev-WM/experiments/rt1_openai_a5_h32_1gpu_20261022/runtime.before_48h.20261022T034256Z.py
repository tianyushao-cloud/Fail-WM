"""Load credentials and proxy from private shared configuration across containers."""
import json
import http.client
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

PRIVATE = Path("/media/tyshao/.config/jev-wm")


def api_environment():
    if not os.environ.get("OPENAI_API_KEY"):
        explicit = os.environ.get("JEV_API_KEY_FILE")
        candidates = [Path(explicit)] if explicit else [PRIVATE / "openai.key", Path("/root/.config/jev-wm/openai.key")]
        key_file = next((p for p in candidates if p.is_file()), None)
        if key_file is None:
            raise RuntimeError("Set OPENAI_API_KEY or JEV_API_KEY_FILE; private shared key file is unavailable")
        key = key_file.read_text().strip()
        if not key:
            raise RuntimeError("The configured private API key file is empty")
        os.environ["OPENAI_API_KEY"] = key
    if "JEV_HTTPS_PROXY" in os.environ:
        proxy = os.environ["JEV_HTTPS_PROXY"]
    else:
        try:
            with socket.create_connection(("127.0.0.1", 17897), timeout=1):
                pass
            proxy = "http://127.0.0.1:17897"
        except OSError:
            config = PRIVATE / "rt1_openai_proxy.json"
            if not config.is_file():
                raise RuntimeError("No local SSH proxy or private shared proxy configuration is available")
            proxy = json.loads(config.read_text())["proxy_url"]
    for name in ["https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"]:
        if proxy:
            os.environ[name] = proxy
        else:
            os.environ.pop(name, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"


def api_health(timeout=12):
    """Check the complete authenticated HTTPS route without buying a score.

    A listening bridge port alone does not establish that its SSH upstream is
    alive. Never put the credential, proxy URL or response body in diagnostics.
    """
    proxy = os.environ.get("HTTPS_PROXY", "")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"https": proxy} if proxy else {}))
    request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            response.read(64)
        return dict(ok=status == 200, status=status, reason=f"HTTP_{status}")
    except urllib.error.HTTPError as error:
        return dict(ok=False, status=error.code, reason=f"HTTP_{error.code}")
    except (OSError, http.client.HTTPException) as error:
        return dict(ok=False, status=None, reason=type(error).__name__)


def wait_for_api(max_wait_seconds=1800, poll_seconds=15, stable_checks=2,
                 on_wait=None):
    """Wait through an SSH reconnect; reject invalid credentials immediately."""
    if max_wait_seconds <= 0 or poll_seconds <= 0 or stable_checks < 1:
        raise ValueError("Network wait settings must be positive")
    started = time.monotonic()
    deadline = started + max_wait_seconds
    successes = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Decisions HTTPS route did not recover within {max_wait_seconds}s; "
                "committed checkpoints and scoring receipts were preserved")
        health = api_health(timeout=min(12, remaining))
        if health["ok"]:
            successes += 1
            if successes >= stable_checks:
                print("Decisions authenticated HTTPS route ready", flush=True)
                return health
            pause = min(2, poll_seconds)
        else:
            successes = 0
            status = health["status"]
            if status is not None and 400 <= status < 500 and status not in (408, 429):
                raise RuntimeError(
                    f"Decisions HTTPS preflight rejected with HTTP {status}; "
                    "check API access and proxy region before launching GPUs")
            record = dict(stage="NETWORK_WAIT", reason=health["reason"],
                          elapsed_seconds=round(time.monotonic() - started),
                          max_wait_seconds=max_wait_seconds)
            if on_wait is not None:
                on_wait(record)
            print("NETWORK_WAIT: " + health["reason"] +
                  "; waiting for the SSH proxy to recover", flush=True)
            pause = poll_seconds
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(pause, remaining))
