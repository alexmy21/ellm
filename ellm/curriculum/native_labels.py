"""Native ring-basis labels and the Phase 5 training curriculum.

At N=1 the candidate set is small: the current working set `S(t)` plus the
previous snapshots. For each turn we collect the exact lattice readings
(Mode B score, door coverage, sizes, ring stats) and a **native soft label**:
the softmax over Mode B scores — the Phase 5 approximation of the ring-basis
soft key `w` (hard key = argmax). The learned policy is trained by soft-label
KL against these native labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..lattice.client import LatticeClient


@dataclass
class TrainingSample:
    """One ranking sample: query + candidates + native labels."""

    query_hex: str
    candidates_hex: list[str]
    features: list[dict]
    native_scores: list[float]
    native_argmax: int

    def prompts(self, prompt_fn) -> list[str]:
        return [prompt_fn(f) for f in self.features]


def softmax(scores: list[float], tau: float = 0.1) -> list[float]:
    """Temperatured softmax over raw scores."""
    if not scores:
        return []
    scaled = [s / max(tau, 1e-6) for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def collect_samples(
    client: LatticeClient,
    turns: list[list[int]],
    ranker,
    gate: str = "ellm",
    tau: float = 0.1,
) -> list[TrainingSample]:
    """Run turns on the lattice and collect (query, candidates, labels)."""
    samples: list[TrainingSample] = []
    history: list[str] = []
    for ids in turns:
        ingest_response = client.ingest(ids)
        state = client.state()
        query_hex = state.get("working_hex") or ""
        if not query_hex:
            continue

        candidates: list[str] = []
        for candidate in [query_hex, *history[-2:]]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        history.append(query_hex)
        if len(candidates) < 2:
            continue

        ring = ingest_response.get("ring") or {}
        features = ranker.features(query_hex, candidates, ring=ring)
        bss = [float(f["bss"]) for f in features]
        native_scores = softmax(bss, tau=tau)
        samples.append(
            TrainingSample(
                query_hex=query_hex,
                candidates_hex=candidates,
                features=features,
                native_scores=native_scores,
                native_argmax=max(range(len(bss)), key=lambda i: bss[i]),
            )
        )
    return samples


def probe_accuracy(
    hub,
    probes: list[list[int]],
    gate: str = "ellm",
    tau: float = 0.1,
) -> float:
    """Held-out probe ranking accuracy of the current policy.

    Runs on a fresh in-memory side-car so probes never touch the shared
    store.
    """
    from ..model.ranker import QwenBroadcastRanker

    if not probes:
        return 1.0
    correct = 0
    total = 0
    with LatticeClient() as client:
        tokens = [f"tid{i}" for i in range(1, 32)]
        client.gate_put(gate, tokens)
        ranker = QwenBroadcastRanker(hub, client, gate=gate)
        samples = collect_samples(client, [list(p) for p in probes], ranker, gate=gate, tau=tau)
        for sample in samples:
            ordered = ranker.discover(sample.query_hex, sample.candidates_hex)
            if not ordered:
                continue
            predicted = sample.candidates_hex.index(ordered[0])
            total += 1
            if predicted == sample.native_argmax:
                correct += 1
    if total == 0:
        return 1.0
    return correct / total
