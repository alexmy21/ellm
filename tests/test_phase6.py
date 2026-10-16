"""Phase 6 integration test: the growth experiment at N=2.

The claim: attaching a second LLM is adding one door endpoint — the eLLM's
recruiter code never changes. Each host runs its own side-car (own gate, own
`H(t)`, own `door.py`); the broadcast collects two door verdicts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ellm.lattice import LatticeClient
from ellm.serve.broadcast import BroadcastDispatcher, NativeStubPolicy
from ellm.serve.doors import DoorNetwork
from ellm.serve.host import HostLLM

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


def _dispatcher(ellm_client: LatticeClient) -> BroadcastDispatcher:
    dispatcher = BroadcastDispatcher(ellm_client, NativeStubPolicy(ellm_client))
    dispatcher.network = DoorNetwork()
    dispatcher.network.attach("ellm", ellm_client)
    return dispatcher


def test_attach_second_door_routes_without_any_change() -> None:
    _ensure_binary()
    ellm_client = LatticeClient()
    host_client = LatticeClient()
    try:
        ellm_client.gate_put("ellm", [f"tid{i}" for i in range(1, 17)])

        dispatcher = _dispatcher(ellm_client)
        ellm_client.ingest([1, 2])

        # N=1 baseline: one door, hire self.
        baseline = dispatcher.select(dispatcher.broadcast(dispatcher.candidates()[0]))
        assert baseline.target == "ellm"

        # Attach the second LLM: the ONLY change is one endpoint.
        host_client.gate_put("host_b", [f"tid{i}" for i in range(17, 33)])
        dispatcher.network.attach("host_b", host_client)

        def route(probe: list[int]):
            ingest_response = ellm_client.ingest(probe)
            verdicts = dispatcher.broadcast(ingest_response["turn_hex"])
            assert len(verdicts) == 2, "the broadcast must collect two door verdicts"
            return dispatcher.select(verdicts)

        # A turn in the eLLM's language hires the eLLM.
        assert route([1, 2]).target == "ellm"
        # A turn in the host's language hires the host — routing, not retraining.
        assert route([17, 18]).target == "host_b"
        # A turn spanning both languages has no full-coverage door -> fallback.
        assert route([1, 17]).fallback is True
    finally:
        ellm_client.close()
        host_client.close()


def test_ceo_escalation_with_overlapping_codebooks() -> None:
    _ensure_binary()
    ellm_client = LatticeClient()
    host_client = LatticeClient()
    try:
        tokens = [f"tid{i}" for i in range(1, 9)]
        ellm_client.gate_put("ellm", tokens)
        host_client.gate_put("host_b", tokens)
        ellm_client.ingest([1, 2])

        network = DoorNetwork({"ellm": ellm_client, "host_b": host_client})
        dispatcher = BroadcastDispatcher(ellm_client, NativeStubPolicy(ellm_client), network=network)

        verdicts = dispatcher.broadcast(dispatcher.candidates()[0])
        assert all(v.can_restore for v in verdicts)

        selection = dispatcher.select(verdicts)
        assert selection.needs_ceo is True
        assert set(selection.needs_user) == {"ellm", "host_b"}
    finally:
        ellm_client.close()
        host_client.close()


def test_host_llm_runs_its_own_travel_loop() -> None:
    _ensure_binary()
    client = LatticeClient()
    try:
        host = HostLLM(
            name="host_b",
            client=client,
            tokens=[f"tid{i}" for i in range(17, 33)],
        )
        result = host.turn([17, 18])
        assert result["name"] == "host_b"
        assert result["context_tokens"] == ["tid17", "tid18"]
        assert result["generated_ids"] == [17, 18, 19]
        assert result["turn"] == 1
    finally:
        client.close()
