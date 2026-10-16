"""Routing-log parsing — the curriculum source from notebooks 13–21.

Notebook 21 (`structural_llm_router.ipynb`) persists one JSON object per
line with the shape:

    {"state": {"bss": …, "drn": …, "ring": …, "windows": …, "crossings": …},
     "options": […], "instructions": …, "decision": …, "confidence": …}

The parser is deliberately tolerant: it keeps the structural fields it
recognizes and preserves unknown keys, so record-shape drift in the notebooks
does not break ingestion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class LatticeStateSnapshot:
    """The structural state of one routing step (bss/drn/ring/windows)."""

    bss: float | None = None
    drn: dict[str, Any] = field(default_factory=dict)
    ring: dict[str, Any] = field(default_factory=dict)
    windows: dict[str, Any] = field(default_factory=dict)
    crossings: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingSample:
    """One `(state, options, decision, confidence)` curriculum item."""

    state: LatticeStateSnapshot
    options: list[Any] = field(default_factory=list)
    decision: str | None = None
    confidence: float | None = None
    instructions: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_state(raw: dict[str, Any]) -> LatticeStateSnapshot:
    known = {"bss", "drn", "ring", "windows", "crossings"}
    snapshot = LatticeStateSnapshot(
        bss=raw.get("bss") if isinstance(raw.get("bss"), (int, float)) else None,
        drn=raw.get("drn") if isinstance(raw.get("drn"), dict) else {},
        ring=raw.get("ring") if isinstance(raw.get("ring"), dict) else {},
        windows=raw.get("windows") if isinstance(raw.get("windows"), dict) else {},
        crossings=raw.get("crossings") if isinstance(raw.get("crossings"), dict) else {},
    )
    snapshot.extra = {k: v for k, v in raw.items() if k not in known}
    return snapshot


def parse_record(record: dict[str, Any]) -> RoutingSample | None:
    """Parse one routing-log record; `None` when the record is unusable."""
    if not isinstance(record, dict):
        return None
    raw_state = record.get("state")
    if not isinstance(raw_state, dict):
        return None

    known = {"state", "options", "decision", "confidence", "instructions"}
    sample = RoutingSample(
        state=parse_state(raw_state),
        options=record.get("options") if isinstance(record.get("options"), list) else [],
        decision=record.get("decision") if isinstance(record.get("decision"), str) else None,
        confidence=record.get("confidence")
        if isinstance(record.get("confidence"), (int, float))
        else None,
        instructions=record.get("instructions")
        if isinstance(record.get("instructions"), str)
        else None,
    )
    sample.extra = {k: v for k, v in record.items() if k not in known}
    return sample


def iter_samples(path: str | Path) -> Iterator[RoutingSample]:
    """Yield parsed samples from a routing_log.jsonl file, skipping bad lines."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = parse_record(record)
            if sample is not None:
                yield sample


def load_samples(path: str | Path) -> list[RoutingSample]:
    return list(iter_samples(path))
