"""Deployable demo integration test: the HTTP door network (two processes).

Starts the host door daemon and the eLLM demo server on ephemeral ports, then
exercises the JSON API exactly as the browser and the containers would.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "lattice" / "target" / "release" / "ellm-lattice"


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _get(url: str) -> dict | str:
    with urllib.request.urlopen(url, timeout=10) as resp:
        body = resp.read().decode()
        return json.loads(body) if resp.headers.get_content_type() == "application/json" else body


@pytest.fixture(scope="module")
def demo_servers(tmp_path_factory):
    if not BINARY.exists():
        subprocess.run(
            ["cargo", "build", "--release", "--offline"],
            cwd=REPO_ROOT / "lattice",
            check=True,
        )

    base = tmp_path_factory.mktemp("ellm-http")
    env_host = {
        **os.environ,
        "ELLM_ROLE": "host",
        "ELLM_NAME": "host_b",
        "ELLM_PORT": "18001",
        "ELLM_REPO": str(base / "host_b"),
        "ELLM_GATE_RANGE": "17,32",
        "ELLM_BINARY": str(BINARY),
    }
    env_ellm = {
        **os.environ,
        "ELLM_ROLE": "ellm",
        "ELLM_NAME": "ellm",
        "ELLM_PORT": "18080",
        "ELLM_REPO": str(base / "ellm"),
        "ELLM_GATE_RANGE": "1,16",
        "ELLM_REMOTE_DOORS": "host_b=http://127.0.0.1:18001",
        "ELLM_BINARY": str(BINARY),
    }

    host_proc = subprocess.Popen(
        [str(REPO_ROOT / ".venv/bin/python"), "-m", "ellm.serve.http"],
        cwd=REPO_ROOT,
        env=env_host,
    )
    ellm_proc = subprocess.Popen(
        [str(REPO_ROOT / ".venv/bin/python"), "-m", "ellm.serve.http"],
        cwd=REPO_ROOT,
        env=env_ellm,
    )
    try:
        for _ in range(50):
            try:
                _get("http://127.0.0.1:18080/api/ping")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("demo server did not start")
        yield "http://127.0.0.1:18080"
    finally:
        ellm_proc.terminate()
        host_proc.terminate()
        ellm_proc.wait(timeout=5)
        host_proc.wait(timeout=5)


def test_browser_page_and_ping(demo_servers) -> None:
    url = demo_servers
    page = _get(url + "/")
    assert "eLLM" in page and "run turn" in page

    ping = _get(url + "/api/ping")
    assert ping == {"name": "ellm", "role": "ellm"}


def test_turn_hires_self_for_ellm_language(demo_servers) -> None:
    result = _post(demo_servers + "/api/turn", {"ids": [1, 2]})
    verdicts = {v["user"]: v for v in result["verdicts"]}
    assert verdicts["ellm"]["can_restore"] is True
    assert verdicts["host_b"]["can_restore"] is False
    assert result["selection"]["target"] == "ellm"
    assert result["generated_ids"] == [1, 2, 3]


def test_turn_routes_to_host_for_host_language(demo_servers) -> None:
    result = _post(demo_servers + "/api/turn", {"ids": [17, 18]})
    verdicts = {v["user"]: v for v in result["verdicts"]}
    assert verdicts["host_b"]["can_restore"] is True
    assert verdicts["ellm"]["can_restore"] is False
    assert result["selection"]["target"] == "host_b"
    # The remote host's door materialized and generated the next turn.
    assert result["context_tokens"] == ["tid17", "tid18"]
    assert result["generated_ids"] == [17, 18, 19]


def test_state_lists_both_doors(demo_servers) -> None:
    state = _get(demo_servers + "/api/state")
    assert set(state["doors"]) == {"ellm", "host_b"}
