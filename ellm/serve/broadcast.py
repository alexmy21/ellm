"""The door-guard broadcast dispatcher — discover → broadcast → select.

The eLLM is the *recruiter*: it sources HLLSet candidates, sends interview
invitations (broadcast), collects door verdicts, and the agent selects:

    * 0 acceptors → fallback (self);
    * 1 acceptor  → hire;
    * >1 acceptors → shortlist to the CEO (the user).

The routing decision is exact — it comes from the doors' coverage verdicts,
never from a learned prediction. The learned `BroadcastPolicy` only decides
which candidates to invite first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..lattice.client import LatticeClient


@dataclass(frozen=True)
class Verdict:
    """One door guard's answer to a broadcast."""

    user: str
    coverage: float
    can_restore: bool


@dataclass(frozen=True)
class Selection:
    """The agent's decision over a set of verdicts."""

    target: str | None = None
    needs_user: tuple[str, ...] = ()
    fallback: bool = False

    @property
    def needs_ceo(self) -> bool:
        return bool(self.needs_user)


class BroadcastPolicy(Protocol):
    """Discovery: order/select which HLLSet candidates to broadcast.

    The policy never names hosts and never decides the winner — the doors do.
    """

    def discover(self, query_hex: str, candidates_hex: list[str]) -> list[str]: ...


class NativeStubPolicy:
    """Mode B BSS-threshold discovery (the deterministic baseline).

    Scores every candidate against the query with `setkey_hex`, keeps the
    ones at or above `theta`, and returns them best-first.
    """

    def __init__(self, client: LatticeClient, theta: float = 0.0):
        self.client = client
        self.theta = theta

    def discover(self, query_hex: str, candidates_hex: list[str]) -> list[str]:
        if not candidates_hex:
            return []
        scores = self.client.setkey_hex(query_hex, candidates_hex)
        ranked = sorted(
            ((score, candidate) for score, candidate in zip(scores, candidates_hex)),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [candidate for score, candidate in ranked if score >= self.theta]


class BroadcastDispatcher:
    """The recruiter: sources candidates, broadcasts them, collects verdicts."""

    def __init__(self, client: LatticeClient, policy: BroadcastPolicy | None = None):
        self.client = client
        self.policy = policy or NativeStubPolicy(client)

    def candidates(self) -> list[str]:
        """Source candidates. At N=1: the eLLM's own working set S(t)."""
        state = self.client.state()
        working_hex = state.get("working_hex") or ""
        return [working_hex] if working_hex else []

    def discover(self, query_hex: str, candidates_hex: list[str]) -> list[str]:
        """Apply the broadcast policy to order/prune candidates."""
        return self.policy.discover(query_hex, candidates_hex)

    def broadcast(self, hllset_hex: str) -> list[Verdict]:
        """Send one candidate to every door; collect verdicts."""
        raw = self.client.door_probe(hllset_hex)
        return [
            Verdict(
                user=v["user"],
                coverage=float(v["coverage"]),
                can_restore=bool(v["can_restore"]),
            )
            for v in raw
        ]

    def select(self, verdicts: list[Verdict]) -> Selection:
        """Decide from the verdicts: hire, fallback, or escalate to the CEO."""
        acceptors = [v for v in verdicts if v.can_restore]
        if not acceptors:
            return Selection(fallback=True)
        if len(acceptors) == 1:
            return Selection(target=acceptors[0].user)
        return Selection(needs_user=tuple(v.user for v in acceptors))

    def run(self, query_hex: str) -> tuple[list[str], list[Verdict], Selection]:
        """One full dispatch turn: discover → broadcast → select."""
        candidates = self.discover(query_hex, self.candidates())
        verdicts: list[Verdict] = []
        for candidate in candidates:
            verdicts.extend(self.broadcast(candidate))
        return candidates, verdicts, self.select(verdicts)
