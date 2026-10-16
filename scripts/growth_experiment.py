#!/usr/bin/env python3
"""Phase 6 growth experiment — attach a second LLM, publish the numbers.

The claim under test: **the eLLM does not change**. Attaching a second LLM is
adding one door endpoint to the network; the recruiter code (discover →
broadcast → verdicts → select) is identical at N=1 and N=2.

Run from the repo root:

    .venv/bin/python scripts/growth_experiment.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "lattice" / "target" / "debug" / "ellm-lattice"


def _ensure_binary() -> None:
    if BINARY.exists():
        return
    subprocess.run(["cargo", "build", "--offline"], cwd=REPO_ROOT / "lattice", check=True)


def main() -> int:
    _ensure_binary()

    from ellm.lattice import LatticeClient
    from ellm.serve.broadcast import BroadcastDispatcher, NativeStubPolicy
    from ellm.serve.doors import DoorNetwork
    from ellm.serve.host import HostLLM

    # ── the eLLM side-car (unchanged since Phase 0) ──────────────────────
    ellm_client = LatticeClient()
    ellm_client.gate_put("ellm", [f"tid{i}" for i in range(1, 17)])

    # ── the recruiter at N=1 (baseline) ──────────────────────────────────
    dispatcher = BroadcastDispatcher(ellm_client, NativeStubPolicy(ellm_client))
    network = DoorNetwork()
    network.attach("ellm", ellm_client)
    dispatcher.network = network

    ellm_client.ingest([1, 2])
    baseline = dispatcher.select(dispatcher.broadcast(dispatcher.candidates()[0]))
    print(f"N=1 baseline: {len(network.doors)} door -> target = {baseline.target}\n")

    # ── attach a second LLM: its own ewm-sm, hllsetLUT, H(t), door.py ────
    host_client = LatticeClient()
    host = HostLLM(
        name="host_b",
        client=host_client,
        tokens=[f"tid{i}" for i in range(17, 33)],  # a *different* codebook
    )
    network.attach("host_b", host_client)
    print(f"N=2 attached : {len(network.doors)} doors; eLLM code unchanged\n")

    # ── the experiment ───────────────────────────────────────────────────
    probes = [[1, 2], [17, 18], [1, 17]]
    print(f"{'probe':<10} | {'ellm coverage':<15} | {'host_b coverage':<16} | selection")
    print("-" * 70)

    rows = []
    for probe in probes:
        ingest_response = ellm_client.ingest(probe)
        candidate = ingest_response["turn_hex"]  # the turn's HLLSet, not the cumulative set
        verdicts = dispatcher.broadcast(candidate)
        selection = dispatcher.select(verdicts)

        by_user = {v.user: v for v in verdicts}
        ellm_cov = by_user.get("ellm").coverage if "ellm" in by_user else float("nan")
        host_cov = by_user.get("host_b").coverage if "host_b" in by_user else float("nan")

        if selection.fallback:
            target = "fallback"
        elif selection.needs_ceo:
            target = "CEO: " + ",".join(selection.needs_user)
        else:
            target = selection.target or "?"
        print(f"{str(probe):<10} | {ellm_cov:<15.3f} | {host_cov:<16.3f} | {target}")

        # The full loop on the selected host: door → generate → ingest.
        if selection.target == "host_b":
            host_result = host.turn(probe)
            loop = f"host_b loop: {host_result['context_tokens']} -> {host_result['generated_ids']}"
        elif selection.target == "ellm":
            data = ellm_client.to_prompt(probe, user="ellm")
            loop = f"ellm loop : {data['tokens']} -> (next turn)"
        else:
            loop = "no door accepted the candidate"
        rows.append((probe, ellm_cov, host_cov, target, loop))
        print(f"{'':<10} | {'':<15} | {'':<16} | {loop}")

    # ── publish the numbers ──────────────────────────────────────────────
    hires = {"ellm": 0, "host_b": 0, "ceo": 0, "fallback": 0}
    for _, _, _, target, _ in rows:
        if target == "ellm":
            hires["ellm"] += 1
        elif target == "host_b":
            hires["host_b"] += 1
        elif target.startswith("CEO"):
            hires["ceo"] += 1
        else:
            hires["fallback"] += 1

    print("\nSummary")
    print(f"  probes      : {len(rows)}")
    print(f"  hired self  : {hires['ellm']}")
    print(f"  hired host  : {hires['host_b']}   (routing to the new LLM, no eLLM change)")
    print(f"  CEO tie     : {hires['ceo']}")
    print(f"  fallback    : {hires['fallback']}")
    print("\nClaim: attaching a second LLM = one `network.attach(...)` call;")
    print("       discover/broadcast/select were never touched.")

    # ── CEO escalation with overlapping codebooks (tie-break) ────────────
    with LatticeClient() as a, LatticeClient() as b:
        a.gate_put("ellm", [f"tid{i}" for i in range(1, 9)])
        b.gate_put("host_b", [f"tid{i}" for i in range(1, 9)])
        a.ingest([1, 2])
        net = DoorNetwork({"ellm": a, "host_b": b})
        d = BroadcastDispatcher(a, NativeStubPolicy(a), network=net)
        verdicts = d.broadcast(d.candidates()[0])
        selection = d.select(verdicts)
        print(f"\nOverlapping codebooks: {selection.needs_ceo=}, shortlist={selection.needs_user}")

    ellm_client.close()
    host_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
