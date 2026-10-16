"""The door network — one `LatticeClient` per LLM side-car.

At N=1 the network has a single door (the eLLM's own side-car). At N=2+ each
LLM attaches its own side-car (own ewm-sm, own `hllsetLUT`, own `H(t)`, own
`door.py`). The broadcast sends the same HLLSet candidate to **every** door;
each answers with its own coverage verdict. Attaching a new LLM is adding one
endpoint — nothing else changes.
"""

from __future__ import annotations

from ..lattice.client import LatticeClient
from .broadcast import Verdict


class DoorNetwork:
    """The broadcast medium: name → side-car client."""

    def __init__(self, doors: dict[str, LatticeClient] | None = None):
        self.doors: dict[str, LatticeClient] = dict(doors or {})

    def attach(self, name: str, client: LatticeClient) -> None:
        """Attach one LLM's side-car (its own gate, H(t), hllsetLUT)."""
        self.doors[name] = client

    def broadcast(self, hllset_hex: str) -> list[Verdict]:
        """Ask every door for its verdict on the candidate."""
        verdicts: list[Verdict] = []
        for name, client in self.doors.items():
            for raw in client.door_probe(hllset_hex, user=name):
                verdicts.append(
                    Verdict(
                        user=raw["user"],
                        coverage=float(raw["coverage"]),
                        can_restore=bool(raw["can_restore"]),
                    )
                )
        return verdicts

    def close(self) -> None:
        for client in self.doors.values():
            client.close()
        self.doors.clear()
