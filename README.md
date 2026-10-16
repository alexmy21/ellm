# ellm — Emerging LLM

**A local LLM with the full travel loop — the first native EWM-SM-aware LLM.**
A small transformer running on the RTX 3060 12 GB that uses
`ewm-state-machine` as its exact, content-addressed memory (the map),
materializes and ingests through the door, generates with attention
(locomotion), and **broadcasts discovered HLLSet candidates to every LLM's
door guard**; each door answers whether it can restore the HLLSet through its
own codebook (`coverage = 1.0`), and the agent selects the target — the user
breaks ties. Candidates are **gated** by each ewm-sm's own `hllsetLUT` (its
registry of generated HLLSets). Its context window is a **managed working set
of lattice nodes** — shrunk and expanded as HLLSets before materialization —
not a linear token buffer.

```text
   parole (host tokens / user prompt)
        │  ingest μ                                   ▲  chosen host generates (parole)
        ▼                                             │
   langue = HLLSet lattice (map) ── materialize ρ ──▶ token context (door)
        │  traversal scores (BSS/Jaccard/overlap)     │  attention (locomotion)
        │  D/R/N · ring · windows                     │
        ▼                                             ▼
   LatticeState ──▶ discover HLLSet candidates ──▶ broadcast to every door
                        │                              │  door verdicts
                        ▼                              ▼
                 select target (0→fallback, 1→route, >1→user)
```

## Lineage — the travel-loop line

```text
  hllset-next-v2        the algebra of record — FULLY INCORPORATED into
        │               ewm-state-machine; kept only as a seed, never a
        │               direct dependency
        ▼
  ewm-state-machine     the state machine: side-car loop, commit DAG,
        │               trajectory, materialisation-to-prompt, router line
        │
        ▼
  ellm                  THE TRAVEL LOOP: lattice memory + local generation
                        + door-guard broadcast dispatcher, on RTX 3060
```

The design follows two publications in
`../ewm-state-machine/REFS/EWM/PUB/`:

* **`attention_lattice_traversal.tex`** — *Attention on Lattice Traversal Is
  All You Need*. The lattice is the map, attention is the locomotion,
  materialization is the door; the exact engine runs in Mode A (address-key),
  Mode B (set-key), and Mode C (hybrid) along one temperature axis.
* **`langue_as_a_lattice.tex`** — *Langue as a Lattice*. The HLLSet lattice is
  a computable langue; the Emerging LLM is the dispatcher — reformulated as
  **door-guard broadcast, not host identification**: discover HLLSet
  candidates, ask every LLM's door guard (`coverage = 1.0`), select the
  target, and let the user break ties.

**Full design:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Stack

| Layer | Technology |
| --- | --- |
| Lattice / state machine | Rust — `ewm-state-machine` crates reused **by path**, unchanged |
| Lattice-native attention | Rust — `ellm-attn`: pure-software exact engine (K-storage, Mode A/B/C) |
| eLLM model + training | Python / PyTorch (`ewm-nanolm` env, torch 2.6.0+cu124) |
| Base model | Qwen2.5-1.5B-Instruct (cached) + LoRA/PEFT adapters |
| GPU | RTX 3060 12 GB (`CUDA_VISIBLE_DEVICES=1`) |

Python talks to the Rust lattice side-car over JSON lines on stdin/stdout —
the same seam as `ewm-state-machine/trainer/ewm.py`.

## Python environment

The project uses a `.venv` created from the `ewm-nanolm` conda Python with
`--system-site-packages`, so GPU torch comes from conda and only the project
deps are installed into the venv:

```bash
uv venv --python /home/alexmy/.conda/envs/ewm-nanolm/bin/python \
    --system-site-packages .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# sanity checks
.venv/bin/python scripts/smoke_gpu.py     # RTX 3060 visible from torch
.venv/bin/python -m pytest tests/ -q      # Python↔Rust seam tests
```

The Rust side is built and tested with:

```bash
cd lattice && cargo build --offline && cargo test --offline
```

## Repository layout

```text
ellm/
├── docs/ARCHITECTURE.md     # the architectural document
├── lattice/                 # Rust side-car workspace (ellm-lattice + ellm-attn)
├── ellm/                    # Python package (lattice client, model, travel, train, serve)
├── notebooks/               # one notebook per phase (00–06)
├── tests/
└── pyproject.toml
```

## Phases

Phases 0–5 run at **N=1** (the eLLM is the only LLM in the network and the
only candidate); phase 6 attaches a second LLM and demonstrates growth.

| Phase | Deliverable | Status |
| --- | --- | --- |
| 0 | repo scaffold: docs, `lattice/` builds, Python package skeleton, GPU smoke test, JSON-lines protocol v0 | done |
| 1 | wire the real lattice: `ellm-lattice` service (ping/ingest/materialize/state/commit/hllset_lut/setkey), `ellm-attn` complete, routing-log ingestion | done |
| 2 | the door: materialize-to-prompt with coverage invariant; the eLLM's own codebook gate (single-LLM case) | done |
| 3 | door-guard broadcast seam at N=1 (degenerate self-selection): Mode B discovery + `door_probe` verdict on the eLLM's own side-car + selection | done |
| 4 | two-process split + closed travel loop at N=1; surprise-gated door policy v0 | next |
| 5 | first learned policy at N=1: LoRA broadcast-policy training (which HLLSets to send first); optional Mode C λ | |
| 6 | growth experiment: attach a second LLM (own ewm-sm + `hllsetLUT` + `H(t)` + `door.py`); broadcast collects two door verdicts with no change to the eLLM | |

## License

MIT — see [LICENSE](LICENSE).
