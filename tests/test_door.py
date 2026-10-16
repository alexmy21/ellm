"""Phase 2 integration test: the door (materialize-to-prompt + coverage guard)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ellm.lattice import LatticeClient
from ellm.travel.door import Door, RoundTripGuardError

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_binary() -> None:
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


def _client_with_gate() -> LatticeClient:
    _ensure_binary()
    client = LatticeClient()
    client.gate_put("ellm", [f"tid{i}" for i in range(1, 9)])
    return client


def test_door_opens_inside_gate() -> None:
    with _client_with_gate() as client:
        door = Door(client, gate="ellm")
        result = door.open([1, 2, 3])
        assert result.refused is False
        assert result.coverage == 1.0
        assert sorted(result.restored_ids) == [1, 2, 3]
        assert door.round_trip([1, 2, 3])


def test_door_refuses_tokens_outside_gate() -> None:
    with _client_with_gate() as client:
        door = Door(client, gate="ellm")
        with pytest.raises(RoundTripGuardError):
            door.open([1, 99])


def test_door_ungated_plain_roundtrip() -> None:
    _ensure_binary()
    with LatticeClient() as client:
        door = Door(client, gate=None)
        result = door.open([7, 8])
        assert result.refused is False
        assert result.restored_ids == [7, 8]
