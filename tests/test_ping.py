"""Phase 0 integration test: Python client ↔ `ellm-lattice` JSON-lines seam."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ellm.lattice import LatticeClient, LatticeError

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_binary() -> None:
    """Build the lattice side-car if the binary is missing."""
    binary = REPO_ROOT / "lattice" / "target" / "debug" / "ellm-lattice"
    if binary.exists():
        return
    result = subprocess.run(
        ["cargo", "build", "--offline"],
        cwd=REPO_ROOT / "lattice",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cargo build failed:\n{result.stderr}")


def test_ping() -> None:
    _ensure_binary()
    with LatticeClient() as client:
        data = client.ping()
        assert data["pong"] is True
        assert "version" in data


def test_setkey_scores() -> None:
    _ensure_binary()
    with LatticeClient() as client:
        scores = client.setkey(["alpha"], [["alpha", "beta"], ["gamma"]])
        assert len(scores) == 2
        assert abs(scores[0] - 1.0) < 1e-9
        assert scores[1] < 1.0


def test_unknown_command_raises() -> None:
    _ensure_binary()
    with LatticeClient() as client:
        with pytest.raises(LatticeError):
            client._call("definitely-not-a-command")
