"""Held-out probes — a frozen set of turns never used for training.

Phase 2 keeps the probe set frozen; Phase 3+ evaluates the ranker against it.
The built-in probe set is synthetic and intentionally tiny.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Frozen, held-out turn samples (`tid{n}` ids). Do not edit after phase 2.
HELD_OUT_PROBES: list[list[int]] = [
    [1, 2, 3],
    [2, 3, 4],
    [10, 11, 12],
]

DEFAULT_PROBE_PATH = Path(__file__).parent / "probes.json"


@dataclass(frozen=True)
class Probe:
    """One frozen probe turn."""

    ids: tuple[int, ...]


def builtin_probes() -> list[Probe]:
    return [Probe(ids=tuple(ids)) for ids in HELD_OUT_PROBES]


def load_probes(path: str | Path | None = None) -> list[Probe]:
    """Load the frozen probe set from JSON, falling back to the built-in set."""
    probe_path = Path(path) if path is not None else DEFAULT_PROBE_PATH
    if probe_path.exists():
        raw = json.loads(probe_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("probes", [])
        return [Probe(ids=tuple(entry)) for entry in raw]
    return builtin_probes()


def save_probes(probes: list[Probe], path: str | Path) -> None:
    """Persist the probe set (used once, when the set is created)."""
    out = [list(p.ids) for p in probes]
    Path(path).write_text(json.dumps({"probes": out}, indent=2), encoding="utf-8")
