"""The trainer process — owns the scheduler, learns WIP, commits at slow time.

Phase 4: the learner is a stub (it bumps the policy generation and keeps the
surprise threshold); the real LoRA learner lands in Phase 5. What is real
already is the **contract**: the trainer publishes a policy only after the
guard set passes, and the server sees it only after publication.
"""

from __future__ import annotations

from typing import Any

from ..lattice.client import LatticeClient


class GuardSet:
    """Minimal acceptance contract for a candidate policy."""

    def __init__(self, accept_min_gain: float = 0.0):
        self.accept_min_gain = accept_min_gain

    def evaluate(self, candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
        # Phase 4 stub: any well-formed candidate passes. Phase 5 replaces
        # this with probe accuracy, regression slack, and progress guards.
        return "generation" in candidate


class Trainer:
    """The trainer process. Publishes the policy at slow time."""

    def __init__(self, client: LatticeClient, schedule_every: int = 2):
        self.client = client
        self.schedule_every = schedule_every
        self.guards = GuardSet()
        self.generation = 0
        self.observations = 0

    def step(self) -> dict[str, Any] | None:
        """Observe one slow-time tick; publish when scheduled and guarded."""
        self.observations += 1
        if self.observations % self.schedule_every != 0:
            return None

        current = self.client.policy_get()
        candidate = self.learn(current)
        if not self.guards.evaluate(candidate, current):
            return None
        self.client.policy_put(candidate)
        self.generation += 1
        return candidate

    def learn(self, current: dict[str, Any] | None) -> dict[str, Any]:
        """WIP learner (stub in Phase 4): bump the generation, keep theta."""
        theta = float((current or {}).get("theta", 0.3))
        return {"generation": self.generation + 1, "theta": theta, "door": "surprise-v0"}
