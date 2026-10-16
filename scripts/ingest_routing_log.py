#!/usr/bin/env python3
"""Ingest `routing_log.jsonl` (notebooks 13–21) into `LatticeState` snapshots.

The real routing_log.jsonl is produced by running notebook 21 in the
`ewm-nanolm` env. Until then, `--demo` writes a small synthetic log with the
same shape so the pipeline is testable end to end.

Usage:

    python scripts/ingest_routing_log.py --demo -o /tmp/snapshots.jsonl
    python scripts/ingest_routing_log.py -i routing_log.jsonl -o snapshots.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ellm.curriculum.routing_log import load_samples


def write_demo_log(path: Path) -> None:
    """Write a synthetic routing_log.jsonl matching the notebook-21 shape."""
    records = [
        {
            "state": {
                "bss": 0.83,
                "drn": {"departed": 2, "retained": 11, "new": 4},
                "ring": {"residual": 3, "in_span": False, "dim": 12},
                "windows": {"short": 0.81, "long": 0.66},
                "crossings": {"bss_short_long": False},
            },
            "options": ["llm_a", "llm_b", "llm_c"],
            "instructions": "route the query to the best host",
            "decision": "llm_a",
            "confidence": 0.91,
        },
        {
            "state": {
                "bss": 0.55,
                "drn": {"departed": 1, "retained": 9, "new": 7},
                "ring": {"residual": 9, "in_span": False, "dim": 14},
                "windows": {"short": 0.62, "long": 0.58},
                "crossings": {"bss_short_long": True},
            },
            "options": ["llm_a", "llm_b", "llm_c"],
            "instructions": "route the query to the best host",
            "decision": "llm_b",
            "confidence": 0.74,
        },
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, help="routing_log.jsonl to ingest")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true", help="write a synthetic log first")
    args = parser.parse_args()

    if args.demo:
        demo_path = args.input or Path("routing_log.demo.jsonl")
        write_demo_log(demo_path)
        args.input = demo_path

    if args.input is None:
        parser.error("provide --input or use --demo")

    samples = load_samples(args.input)
    if not samples:
        print(f"no usable samples in {args.input}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict()) + "\n")

    print(f"ingested {len(samples)} samples: {args.input} -> {args.output}")
    first = samples[0]
    print(
        "first sample: "
        f"bss={first.state.bss} drn={first.state.drn} "
        f"decision={first.decision} confidence={first.confidence}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
