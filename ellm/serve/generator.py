"""Generation inside the materialized token context.

Phase 4 runs the closed travel loop with a deterministic stub generator; the
real Qwen2.5-1.5B generator lands in Phase 5 behind the same `Generator`
protocol.
"""

from __future__ import annotations

from typing import Protocol


class Generator(Protocol):
    """Turn token ids (context) into the next turn's token ids."""

    def generate(self, context_tokens: list[str], policy: dict | None = None) -> list[int]: ...


class StubGenerator:
    """Deterministic native-stub generator.

    Echoes the materialized `tid{n}` context and appends the next id — enough
    to close the travel loop at N=1 without loading a model.
    """

    def generate(self, context_tokens: list[str], policy: dict | None = None) -> list[int]:
        ids = [int(t[3:]) for t in context_tokens if t.startswith("tid")]
        if not ids:
            return [1]
        return [*ids, max(ids) + 1]
