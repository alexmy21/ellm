"""The server process — broadcast + generate + selection, never blocks.

The closed travel loop at N=1:

    1  ingest      the query turn (commits per turn)
    2  hot-swap    re-read the committed policy from the store
    3  surprise    the door policy v0: open the door on novelty spikes
    4  broadcast   discover candidates → door verdicts → select (Phase 3)
    5  generate    chosen speaker generates inside the materialized context
    6  ingest      generated ids become the next turn

The server never schedules and never blocks; it serves one request at a time
and refreshes the policy between requests (hot-swap). The trainer publishes
the policy at slow time; the server sees only committed policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..lattice.client import LatticeClient
from .broadcast import BroadcastDispatcher, Selection, Verdict
from .generator import Generator, StubGenerator
from .surprise import DoorDecision, SurpriseGate

DEFAULT_POLICY: dict[str, Any] = {"theta": 0.3, "door": "surprise-v0"}


@dataclass
class TurnResult:
    """One closed-loop turn."""

    ids: list[int]
    turn: int
    policy: dict[str, Any] | None
    door: DoorDecision
    candidates: list[str] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    selection: Selection | None = None
    context_tokens: list[str] = field(default_factory=list)
    generated_ids: list[int] = field(default_factory=list)


class TravelServer:
    """The N=1 server process."""

    def __init__(
        self,
        client: LatticeClient,
        gate: str = "ellm",
        generator: Generator | None = None,
        surprise: SurpriseGate | None = None,
        dispatcher: BroadcastDispatcher | None = None,
    ):
        self.client = client
        self.gate = gate
        self.generator = generator or StubGenerator()
        self.surprise = surprise or SurpriseGate()
        self.dispatcher = dispatcher or BroadcastDispatcher(client)
        self.policy: dict[str, Any] | None = None

    def hot_swap(self) -> dict[str, Any] | None:
        """Read the committed policy from the store (the trainer's write)."""
        self.policy = self.client.policy_get() or dict(DEFAULT_POLICY)
        return self.policy

    def turn(self, ids: list[int]) -> TurnResult:
        """Serve one travel-loop turn."""
        # 1. ingest — the query turn (commits per turn; skips no-change passes).
        ingest_response = self.client.ingest(ids)

        # 2. hot-swap — the server sees only committed policies.
        self.hot_swap()

        # 3. surprise-gated door policy v0 (per-turn signal).
        state = self.client.state()
        door = self.surprise.decide(ingest_response, state, self.policy)

        # 4. materialize-to-prompt (open the door) or keep the compressed node.
        if door.open_door:
            data = self.client.to_prompt(ids, user=self.gate)
            context_tokens = data.get("tokens", [])
        else:
            context_tokens = ["<compressed>", f"working:{state.get('working_hex', '')[:16]}"]

        # 5. broadcast → select (the recruiter seam).
        candidates = self.dispatcher.candidates()
        verdicts: list[Verdict] = []
        for candidate in candidates:
            verdicts.extend(self.dispatcher.broadcast(candidate))
        selection = self.dispatcher.select(verdicts)

        # 6. generate inside the chosen context (N=1: self).
        generated_ids = self.generator.generate(context_tokens, self.policy)

        return TurnResult(
            ids=ids,
            turn=int(state.get("turn", 0)),
            policy=self.policy,
            door=door,
            candidates=candidates,
            verdicts=verdicts,
            selection=selection,
            context_tokens=context_tokens,
            generated_ids=generated_ids,
        )

    def serve(self, turns: list[list[int]]) -> list[TurnResult]:
        """Serve a batch of turns; the next turn's query is the generated ids."""
        results: list[TurnResult] = []
        query = turns[0] if turns else []
        for i, _ in enumerate(turns):
            result = self.turn(query)
            results.append(result)
            query = result.generated_ids or (turns[i + 1] if i + 1 < len(turns) else [])
        return results
