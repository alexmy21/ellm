"""The door — materialize-to-prompt with the round-trip coverage invariant.

Phase 2: at N=1 the eLLM's own codebook gate (`user="ellm"`) guards the door.
`Door.open` refuses to cross when `coverage < 1.0` (RoundTripGuard).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lattice.client import LatticeClient


class RoundTripGuardError(RuntimeError):
    """Raised when the door refuses a materialization (coverage < 1.0)."""


@dataclass
class DoorResult:
    """One door crossing: the restored token ids and the coverage reading."""

    restored_ids: list[int]
    tokens: list[str]
    coverage: float
    gate: str | None
    refused: bool = False

    def as_prompt_tokens(self) -> list[str]:
        """The token context the LLM attends to (materialize-to-prompt)."""
        return self.tokens


class Door:
    """The materialize/ingest door of the travel loop."""

    def __init__(self, client: LatticeClient, gate: str | None = "ellm"):
        self.client = client
        self.gate = gate

    def open(self, ids: list[int], gate: str | None = None) -> DoorResult:
        """Materialize a turn of `tid{n}` ids into a token context.

        Crosses the door through the configured codebook gate; raises
        [`RoundTripGuardError`] when the round-trip coverage invariant fails.
        """
        data = self.client.to_prompt(ids, user=gate if gate is not None else self.gate)
        if data.get("refused"):
            raise RoundTripGuardError(
                f"door refused: coverage={data.get('coverage')} "
                f"gate={data.get('gate')} ({data.get('reason', '')})"
            )
        result = DoorResult(
            restored_ids=data["restored_ids"],
            tokens=data["tokens"],
            coverage=data["coverage"],
            gate=None if data.get("gate") is None else str(data["gate"]),
            refused=False,
        )
        return result

    def round_trip(self, ids: list[int]) -> bool:
        """Ingest is implicit in `open`; the round trip is faithful iff the
        restored ids are exactly the input ids."""
        result = self.open(ids)
        return result.restored_ids == ids
