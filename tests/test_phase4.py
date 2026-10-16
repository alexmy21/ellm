"""Phase 4 integration test: two processes, one store, closed travel loop."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ellm.lattice import LatticeClient
from ellm.serve.server import TravelServer
from ellm.train.trainer import Trainer

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


def _repo_path(tmp_path: Path) -> str:
    return str(tmp_path / "ewm-repo")


def _seeded_client(tmp_path: Path) -> LatticeClient:
    _ensure_binary()
    client = LatticeClient(repo_path=_repo_path(tmp_path))
    client.gate_put("ellm", [f"tid{i}" for i in range(1, 16)])
    return client


def test_two_processes_share_one_store(tmp_path: Path) -> None:
    _ensure_binary()
    trainer_client = LatticeClient(repo_path=_repo_path(tmp_path))
    server_client = LatticeClient(repo_path=_repo_path(tmp_path))
    try:
        trainer = Trainer(trainer_client, schedule_every=1)

        assert server_client.policy_get() is None

        candidate = trainer.step()
        assert candidate is not None and candidate["generation"] == 1

        # The server process sees the published policy without restart.
        assert server_client.policy_get() == candidate
    finally:
        trainer_client.close()
        server_client.close()


def test_closed_loop_turn_at_n1(tmp_path: Path) -> None:
    with _seeded_client(tmp_path) as client:
        server = TravelServer(client, gate="ellm")
        result = server.turn([1, 2])

        assert result.turn == 1
        assert result.selection is not None and result.selection.target == "ellm"
        assert result.generated_ids, "the loop must produce the next turn"
        assert result.policy is not None


def test_surprise_gate_opens_then_keeps_compressed(tmp_path: Path) -> None:
    with _seeded_client(tmp_path) as client:
        server = TravelServer(client, gate="ellm")

        first = server.turn([1, 2])
        assert first.door.open_door is True, "first turn is all new bits"

        second = server.turn([1, 2])
        assert second.door.open_door is False, "no new bits: keep the compressed node"
        assert "<compressed>" in second.context_tokens


def test_policy_hot_swap_between_turns(tmp_path: Path) -> None:
    _ensure_binary()
    trainer_client = LatticeClient(repo_path=_repo_path(tmp_path))
    server_client = LatticeClient(repo_path=_repo_path(tmp_path))
    try:
        server_client.gate_put("ellm", [f"tid{i}" for i in range(1, 16)])
        trainer = Trainer(trainer_client, schedule_every=1)
        server = TravelServer(server_client, gate="ellm")

        before = server.turn([1, 2])
        assert before.policy is None or before.policy.get("generation", 0) == 0

        published = trainer.step()
        assert published is not None

        after = server.turn([3, 4])
        assert after.policy is not None
        assert after.policy["generation"] == published["generation"]
    finally:
        trainer_client.close()
        server_client.close()
