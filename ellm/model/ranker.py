"""The learned broadcast policy — LoRA on Qwen2.5-1.5B.

Implements `BroadcastPolicy.discover`: for each candidate HLLSet, build a
deterministic lattice-state prompt (query bits, candidate bits, Mode B score,
door coverage, ring stats), read the Qwen last hidden state, and score it
with the learned head. The ranking order is the recruiter's invitation order;
the doors still decide the hire.
"""

from __future__ import annotations

from typing import Protocol

from ..lattice.client import LatticeClient
from .base import QwenHub

_PROMPT = (
    "query_bits={query_bits} candidate_bits={candidate_bits} "
    "bss={bss:.4f} coverage={coverage:.4f} "
    "residual={residual:.4f} in_span={in_span} dimension={dimension}"
)


class LatticeFeatureSource(Protocol):
    """Computes per-candidate lattice features for a query."""

    def features(self, query_hex: str, candidates_hex: list[str]) -> list[dict]: ...


class QwenBroadcastRanker:
    """The recruiter policy: Qwen + LoRA + score head over lattice features."""

    def __init__(
        self,
        hub: QwenHub,
        client: LatticeClient,
        gate: str = "ellm",
        ring: dict | None = None,
    ):
        self.hub = hub
        self.client = client
        self.gate = gate
        self.ring = ring or {}

    def features(
        self,
        query_hex: str,
        candidates_hex: list[str],
        ring: dict | None = None,
    ) -> list[dict]:
        """Exact lattice readings: Mode B score, door coverage, sizes, ring."""
        if not candidates_hex:
            return []
        ring = ring if ring is not None else self.ring
        bss = self.client.setkey_hex(query_hex, candidates_hex)
        pops = self.client.popcount_hex(candidates_hex)
        query_bits = self.client.popcount_hex([query_hex])[0]
        coverages = [
            self.client.door_probe(c, user=self.gate)[0]["coverage"] for c in candidates_hex
        ]
        features: list[dict] = []
        for i, candidate in enumerate(candidates_hex):
            features.append(
                {
                    "query_hex": query_hex,
                    "candidate_hex": candidate,
                    "query_bits": query_bits,
                    "candidate_bits": int(pops[i]),
                    "bss": float(bss[i]),
                    "coverage": float(coverages[i]),
                    "residual": float(ring.get("residual", 0.0)),
                    "in_span": int(ring.get("in_span", 0)),
                    "dimension": int(ring.get("dimension", 0)),
                }
            )
        return features

    def _query_bits(self, query_hex: str) -> int:
        pops = self.client.popcount_hex([query_hex])
        return int(pops[0])

    def prompt(self, features: dict) -> str:
        return _PROMPT.format(**features)

    def score(self, features_list: list[dict]) -> list[float]:
        """Score a batch of candidate features (no gradient)."""
        prompts = [self.prompt(f) for f in features_list]
        logits = self.hub.score(prompts)
        return [float(x) for x in logits.squeeze(-1).tolist()]

    def discover(self, query_hex: str, candidates_hex: list[str]) -> list[str]:
        """Order candidates by learned score, best first (the invitation order)."""
        if not candidates_hex:
            return []
        features = self.features(query_hex, candidates_hex)
        scores = self.score(features)
        ranked = sorted(zip(scores, candidates_hex), key=lambda p: p[0], reverse=True)
        return [candidate for _, candidate in ranked]
