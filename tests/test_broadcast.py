"""Phase 3 integration test: the door-guard broadcast seam at N=1."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ellm.lattice import LatticeClient
from ellm.serve.broadcast import BroadcastDispatcher, NativeStubPolicy

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


def test_broadcast_hires_single_acceptor_self() -> None:
    _ensure_binary()
    with LatticeClient() as client:
        client.gate_put("ellm", [f"tid{i}" for i in range(1, 6)])
        client.ingest([1, 2])

        dispatcher = BroadcastDispatcher(client, NativeStubPolicy(client))
        candidates = dispatcher.candidates()
        assert len(candidates) == 1

        query_hex = candidates[0]
        verdicts = dispatcher.broadcast(query_hex)
        assert len(verdicts) == 1
        assert verdicts[0].can_restore is True
        assert verdicts[0].coverage == 1.0

        selection = dispatcher.select(verdicts)
        assert selection.target == "ellm"
        assert selection.fallback is False
        assert selection.needs_ceo is False


def test_broadcast_falls_back_when_no_door_accepts() -> None:
    _ensure_binary()
    with LatticeClient() as client:
        client.gate_put("ellm", ["tid1", "tid2"])
        client.ingest([1, 2, 99])  # tid99 is outside the gate

        dispatcher = BroadcastDispatcher(client, NativeStubPolicy(client))
        candidates = dispatcher.candidates()
        verdicts = dispatcher.broadcast(candidates[0])
        assert verdicts[0].can_restore is False

        selection = dispatcher.select(verdicts)
        assert selection.fallback is True
        assert selection.target is None


def test_broadcast_multiple_acceptors_escalate_to_user() -> None:
    _ensure_binary()
    with LatticeClient() as client:
        tokens = [f"tid{i}" for i in range(1, 6)]
        client.gate_put("ellm", tokens)
        client.gate_put("host_b", tokens)
        client.ingest([1, 2])

        dispatcher = BroadcastDispatcher(client, NativeStubPolicy(client))
        verdicts = dispatcher.broadcast(dispatcher.candidates()[0])
        assert len(verdicts) == 2
        assert all(v.can_restore for v in verdicts)

        selection = dispatcher.select(verdicts)
        assert selection.target is None
        assert selection.needs_ceo is True
        assert set(selection.needs_user) == {"ellm", "host_b"}
