"""The surprise-gated door policy v0.

The attention paper's first open question: open the door (materialize the
full token context) when novelty or ring residual spikes; otherwise keep the
compressed lattice node.

v0 reads the per-turn `ingest` response:
* no commit → no new bits → keep the compressed node;
* commit present → novelty = `dnr.new / working_bits`; open the door when
  `novelty >= theta` (or the ring residual is non-zero).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoorDecision:
    open_door: bool
    novelty: float
    reason: str


class SurpriseGate:
    """`open_door` iff the turn committed and its novelty clears `theta`."""

    def __init__(self, theta: float = 0.3):
        self.theta = theta

    def decide(
        self,
        ingest_response: dict,
        state: dict,
        policy: dict | None = None,
    ) -> DoorDecision:
        theta = float((policy or {}).get("theta", self.theta))

        if ingest_response.get("commit") is None:
            return DoorDecision(
                open_door=False,
                novelty=0.0,
                reason="no new bits (no commit): keep the compressed node",
            )

        dnr = ingest_response.get("dnr") or {}
        new_bits = float(dnr.get("new") or 0.0)
        working = state.get("working_popcounts") or [0, 0, 0]
        total = float(sum(int(b) for b in working))
        novelty = new_bits / total if total > 0 else 0.0
        residual = float((ingest_response.get("ring") or {}).get("residual") or 0.0)

        if novelty >= theta or residual > 0.0:
            return DoorDecision(
                open_door=True,
                novelty=novelty,
                reason=f"surprise: novelty {novelty:.3f} >= {theta} (residual {residual})",
            )
        return DoorDecision(
            open_door=False,
            novelty=novelty,
            reason=f"novelty {novelty:.3f} < {theta}: keep the compressed node",
        )
