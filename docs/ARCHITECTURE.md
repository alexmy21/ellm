# eLLM — Emerging LLM Architecture

> **Status:** architecture definition + **Phase 0–6 complete** + **deployable demo** (two-container podman-compose network, shared lattice volume, browser UI, optional GPU passthrough), 2026-10-15.
> **Scope:** a **local LLM with the full travel loop** — a small transformer that
> runs on the RTX 3060 12 GB, uses `ewm-state-machine` as its exact,
> content-addressed memory (the map), materializes/ingests through the door,
> generates with attention (locomotion), and carries a **door-guard broadcast
> dispatcher** that discovers lattice HLLSet candidates, asks every LLM's door
> who can restore them, and lets an agent select the speaker.
> **Lineage:** `hllset-next-v2` → `ewm-state-machine` → **`ellm`** (this repo).
> `hllset-next-v2` is **fully incorporated** into `ewm-state-machine`: its
> foundation crates (`hllset-*`, `context-tree`) now live there and are the
> only API the eLLM uses. The old repo is kept only as a **seed for future
> developments** — never a direct dependency.
> **Predecessor:** `../emerging-llm` (dispatcher-only stub line; its two-process
> split, `DecisionRouter` seam, curriculum, guards, and checkpoint discipline are
> absorbed here).

---

## 0. The idea in one paragraph

The two publications define the eLLM as the closure of two claims.

* **`attention_lattice_traversal.tex`** — the lattice is the **map**, attention
  is the **locomotion**, materialization is the **door**, and the seam between
  them is one exact lemma: the support of an attention layer is the lattice join
  of the attended value nodes, so *the bitmap fixes the working set and the
  weights decide the move*. Every transformer device has a lattice counterpart
  (heads = worlds, `W_O` = bridge, depth = complement traversal with a Noether
  stop, causal mask = monotone ring basis, KV cache = commit DAG, residual
  stream = union invariant + `D/R/N`, Q/K/V = bootstrap permutation). The exact
  engine runs in three modes — **A** address-key, **B** set-key, **C** hybrid —
  along one temperature axis.
* **`langue_as_a_lattice.tex`** — the HLLSet lattice is a **computable langue**:
  ingestion `μ` compiles any host language into the shared 32 768-bit plane;
  `D/R/N` are the syntagmatic step, directed BSS reads succession, the GF(2)
  ring basis is the system of distinctive oppositions, and
  `(ingest, materialize)` is the sign governed by the coverage invariant. The
  **Emerging LLM is the dispatcher** — but formulated as **door-guard
  broadcast, not host identification**: the eLLM discovers candidate HLLSets
  and broadcasts them; every LLM's door guard answers whether it can restore
  them (`coverage = 1.0`); the agent selects the target and the user breaks
  ties. *Langue does not speak; parole does; the doors say who can speak.*

**The eLLM is both at once.** It is a small local LLM that *generates*
(locomotion inside a materialized node), *remembers* exactly through the EWM
state machine (the map), *crosses* the door in both directions
(materialize → generate → ingest), and *broadcasts* discovered lattice HLLSets
to every LLM's door guard — including its own — so a thin agent can select the
speaker from the door verdicts. It is the travel loop closed around a
broadcast dispatcher.

```text
   parole (host tokens / user prompt)
        │  ingest μ                                   ▲  chosen host generates (parole)
        ▼                                             │
   langue = HLLSet lattice (map) ── materialize ρ ──▶ token context (door)
        │  traversal scores (BSS/Jaccard/overlap)     │  attention (locomotion)
        │  D/R/N · ring · windows                     │
        ▼                                             ▼
   LatticeState ──▶ discover HLLSet candidates ──▶ broadcast to every door
                        │                             │  door verdicts
                        ▼                             ▼
                 select target (0→fallback, 1→route, >1→user)
```

---

## 1. Positioning

| Project | Adds | eLLM's dependency |
| --- | --- | --- |
| `hllset-next-v2` | the algebra of record — **fully incorporated into `ewm-state-machine`** (its `hllset-*` / `context-tree` crates); kept as a seed for future developments | **no direct dependency** |
| `ewm-state-machine` | state `H(t)=(S(t),H(t-1),D,R,N)`, commit DAG, ring basis, side-car loop | **reused by path, unchanged** |
| `emerging-llm` | dispatcher architecture: two processes, curriculum, guards, checkpoints | absorbed as design (Python implementation) |
| **`ellm`** | **the travel loop + door-guard broadcast + local generation, trained on the RTX 3060** | — |

The eLLM does **not** train tokenizers, does **not** name hosts (it asks their
doors), and does **not** mutate the lattice. Host encodings are inputs;
the lattice is a read-mostly store; only policy updates and state commits are
written.

---

## 2. Conceptual architecture: map, door, move, and actualization

Four roles, each with an owner:

| Role | Paper concept | Owner |
| --- | --- | --- |
| **map** | HLLSet lattice; node = waypoint with detail compressed inside | `ewm-state-machine` (Rust, unchanged) |
| **door** | `ingest` / `materialize`; round-trip law guarded by coverage = 1 | `hllset-morphisms` + lattice side-car |
| **move** | attention inside the materialized token context; Modes A/B/C | base LLM (PyTorch) + `ellm-attn` (pure software) |
| **actualization** | door-guard broadcast: discover HLLSet candidates → broadcast → each door answers `{coverage, can_restore}` → select (user breaks ties) | eLLM discovery + every side-car's door guard + agent selection |

The **travel loop** (from the attention paper, Figure "the travel loop") is the
top-level control flow:

```text
   lattice (map) ──materialize──▶ token context ──attention──▶ move
        ▲                                                           │
        └──────────── commit ◀── ingest ◀── generated tokens ◀───┘
```

**Restoration is two-stage, and the eLLM only ever runs stage 1.** When the
eLLM "passes an HLLSet to a host LLM", it is actually passing it to that
host's **side-car**:

```text
   stage 1 (side-car, EWM side)          stage 2 (inside the host LLM)
   HLLSet ──materialize──▶ tid{n} ids ──▶ original host tokens
           (exact: coverage = 1.0)        (host's own inverse codebook lookup)
```

* **Stage 1** restores the host's LLM **encodings** — `tid{n}` ids, which are
  "tokens" from the EWM point of view. This stage is lattice-exact: the door
  guard's `coverage = 1.0` makes it lossless.
* **Stage 2** restores the host's **original tokens** from those encodings.
  This mapping is the host LLM's private codebook/embedding inverse — the
  `tid{n}` id selects a shared codebook anchor (`CodebookEncoder::from_anchors`,
  bit-identical on both sides), the host reconstructs its latent vector from
  that anchor, and maps it back to its own tokens. It stays entirely inside
  the host.

**Dense embeddings are never touched by the eLLM or the lattice.** The host
LLM manages its embeddings; its side-car only ever receives the *quantized*
stream (`tid{n}` ids) — the dense vectors cross the boundary only at the
instant of quantization and are never stored, learned, or operated on. That
is why the lattice representation is complete by itself, and why the labels
can be **native**: with HLLSets as fixed-size bit-vectors (32 768 bits over
GF(2)), the ring basis (RREF; soft key `w`, hard key `c`, spill `s`) is the
natural coordinate frame — no borrowed, external labels are needed.

The **dispatcher (door-guard broadcast)** sits between the map and the move:
after the door opens, it discovers lattice HLLSet candidates and broadcasts
them to every LLM's door guard; each door answers whether it can restore the
HLLSet through its own codebook; the agent picks the speaker from the
verdicts (self or a host LLM). When only *self* accepts, the loop is a
conventional local LLM with lattice memory; when a *host* accepts, the eLLM
acts as the langue-level router of `langue_as_a_lattice.tex`.

### 2.1 Users, worlds, and the shared lattice

The multi-user foundation (from the companion paper's world model, as used in
the side-car notebooks):

1. **Each user runs its own EWM State Machine.** Every expert / perceptron /
   host LLM — and the eLLM itself — owns a private state machine: its own
   `H(t) = (S(t), H(t-1), D, R, N)`, its own LUTs, its own codebook gate
   `G_u`. No user reads another user's LUT directly.
2. **One shared lattice.** All users push HLLSets into the **same** Boolean
   lattice over the soldered plane: `P = 10`, `M = 1024`, `b = 32`,
   `TOTAL_BITS = 32 768` (from `hllset-contracts`). Because every HLLSet is a
   subset of that one plane, join/meet/difference across users are exact bit
   operations — the "one universe". The union `S(t) = ∪ S_i(t)` and the top
   HLLSet over all hosts' `H(t)` are views over that shared lattice.
3. **Different hashes = different worlds.** A world is `(h, s, τ)` — hash
   type/seed and threshold. Users may use different hash configurations; this
   changes where their tokens land on the plane (their measurement) and hence
   their interpretation fibres (their LUTs). **The geometry `P, b` is shared;
   the hash is not.** Two worlds communicate only through the bridge
   (`ingest ∘ transcribe ∘ materialize`).
4. **The eLLM learns on the state machine's hashed tokens.** The eLLM's input
   vocabulary is the ewm-sm encoding stream, not raw text: host encodings →
   `CodebookEncoder` → `tid{n}` ids → the state machine's ingest hashes those
   tid bytes (murmur3, per-world seed) into bit addresses. `tid{n}` is the
   **side-car's inscription** and the only encoding compatible with the
   soldered lattice geometry (`P=10`, `b=32`). Host encodings and their EWM
   hash values are **isomorphic up to hash collision** — algebraically equal
   for the lattice, with the hash the technically consistent form: fixed-size,
   deterministic, host-agnostic. The eLLM then
   learns on the **lattice image of the encodings**: bit addresses for Mode A
   (the learned key table), HLLSets for Mode B/C, and the lattice state
   (`bss`, `D/R/N`, ring, windows) for the dispatcher. Tokenizers and host
   vocabularies are inputs, never trained.

In short: **many state machines, one plane, per-user interpretations, and a
dispatcher that learns on the hashed encodings flowing through them.**

### 2.2 The solution: the eLLM is the first native EWM-SM-aware LLM

The eLLM is not a host LLM with a side-car bolted on. It is the **first LLM
whose own context management is the HLLSet lattice**. This is the key
asymmetry:

* **Host LLMs** see the lattice only *indirectly*: the door materializes a
  node into a token prompt for them, and their own context window is the
  ordinary linear buffer. They benefit from the map, but they do not manage
  it.
* **The eLLM** manages its context **as lattice nodes, natively**:
  * its **context window is a managed working set** — *shrink* (meet with the
    query, restrict to a sub-lattice, minimal cover) and *expand* (join
    neighbors, lower the BSS threshold, add bootstrap n-gram views) happen on
    HLLSets **before** materialization;
  * its **KV cache is the commit DAG**, its **residual stream is the union
    invariant** with the `D/R/N` write/keep/drop reading;
  * it commits its own generations as new nodes and materializes its own
    context on demand (the surprise-gated door);
  * its attention can be steered by the exact engine (Mode B lookup, Mode C
    hybrid) because it reads the same plane it writes to.

The travel loop is therefore not a wrapper around an LLM — it is **the eLLM's
native runtime**. The broadcast dispatcher is the one piece that reaches
beyond its own loop: it discovers HLLSets in the shared lattice, asks every
door who can restore them, and the agent decides who speaks next, itself
included. That is the "solution" the two papers converge on: *langue* (the
lattice) is not an external memory for the eLLM; it is the eLLM's context,
and *parole* (actualization) is its executive act, grounded in exact door
verdicts.

### 2.3 Dispatch as door-guard broadcast — ask every LLM, trust the doors

The dispatch task is reformulated twice over. First: the eLLM never names
hosts — it discovers **HLLSet candidates**. Second: the routing decision is
not predicted — it is **asked**. Every LLM side-car (including the eLLM's)
runs the same `door.py`, and the door guard already answers the one exact
question that matters: *can this LLM materialize this HLLSet through its own
codebook gate with coverage = 1.0?* So:

```text
   discover  →  broadcast  →  door verdicts  →  select
   [eLLM]       [network]     [every side-car]  [agent]

   D : Lat × State → {HLLSet candidates}                       [eLLM, traversal]
   Broadcast : candidates → all connected LLM side-cars        [network]
   Guard_i : HLLSet → {coverage, can_restore}                  [each LLM's door]
   Select : verdicts → target LLM | fallback | user choice     [agent]
```

1. **Stage 0 — gate (the per-ewm-sm `hllsetLUT`).** Every ewm-sm side-car
   (one per LLM, including the eLLM) manages its own `hllsetLUT` — the
   append-only, touch-counted registry of **all HLLSets that ewm-sm
   generated** (`<name, SHA1, TH>` in `hllset-morphisms::HllsetLut`;
   `<ObjectId, TH>` in `ewm-git::HllsetLut`). The candidate set is collected
   **from these registries**, so HLLSets irrelevant to any host never enter
   the broadcast. The gate is membership: a candidate must be registered
   (`th > 0`) in some host's `hllsetLUT`, and the `ranked()` view (TH
   order) is the natural recency/relevance prior.
2. **Stage 1 — discover (the eLLM).** Given the query HLLSet `Q = I(query)`
   and the gated candidate set (hosts' `H(t)` nodes and their sub-nodes in
   the commit DAG), the eLLM selects **which HLLSet(s) to broadcast** with
   the lattice morphisms (`bss_coverage`, `jaccard`, `overlap` — Mode B),
   threshold `θ`, shrink/expand. The learned part is here: which candidates,
   in which order. This is the traversal of `attention_lattice_traversal.tex`
   Definition 1 — *select by σ, accumulate by join*.
3. **Stage 2 — broadcast.** The candidate HLLSet(s) go to **every connected
   LLM side-car**. No host registry: the network is whatever is attached and
   answering.
4. **Stage 3 — door verdicts.** Each side-car's door guard probes the HLLSet
   against its **own codebook gate** (flat n-seed coverage) and answers
   `{coverage, can_restore}` — the exact `RoundTripGuard` already implemented
   in `door.py`. A door that accepts is *capable* of materializing the HLLSet
   through its own vocabulary. Optional sophistication: each door may also
   report ring/trajectory confirmation (residual against its own basis,
   `D/R/N` consistency with its own tip) alongside coverage.
5. **Stage 4 — select (the agent).**
   * **0 acceptors** → fallback (self) or refusal;
   * **1 acceptor** → route to it;
   * **>1 acceptors** → the user chooses (or a learned default by
     coverage/confidence). This is the architecture's existing rule: when two
     hosts' contexts are equally valid, only the user can choose.

What this buys:

* **The exact decision is the doors'.** Routing correctness is `coverage =
  1.0`, computed by each host's own guard — not a learned prediction. The
  ranker only discovers candidates and orders the broadcast.
* **Host-LLM agnostic by construction.** Attaching an LLM = its side-car
  starts answering broadcasts. No registry, no retraining, no owner lookup.
* **Self is not special-cased.** The eLLM's own door is one of the
  responders; if it accepts, self is a candidate like any other.
* **Recognition without production, distributed.** The langue paper's
  dispatcher asymmetry is now realized by every door: each host recognizes
  whether the HLLSet is in its langue (coverage) without producing anything.
* **Ambiguity surfaces to the user.** Multiple acceptors means genuine
  equivalence in the lattice; the tie-break is human, as designed.

In one line: **the eLLM discovers HLLSets and broadcasts them; the doors say
who can restore them; the agent selects — and the user breaks ties.**

**The recruiter reading.** The eLLM is a *recruiter*: it sources candidates
(the lattice `H(t)` nodes), screens résumés (the `hllsetLUT` gate), and sends
interview invitations (the broadcast). Each LLM's door guard is the
*interview*: "can you actually restore this HLLSet through your own
codebook?" — `coverage = 1.0` is a yes. One yes → hire. No yes → fallback
(self). Several yeses → the shortlist goes to the *CEO* (the user), who makes
the final choice. **Training teaches the recruiter skills** — which candidates
to invite first, which to skip, how to order the interviews — never who to
hire: hiring is decided by the interviews, and ties by the CEO.

### 2.4 Start from a network of one

The system is built to **start with a single LLM and grow by attachment**:

* **Network of 1.** Initially the only LLM in the network is the eLLM itself,
  running its own ewm-sm side-car with its own LUTs, its own `hllsetLUT`, and
  its own `H(t)`. The candidate set has exactly one member — the eLLM's
  `H(t)` — so the full pipeline runs **degenerate but real**:
  * the gate passes the eLLM's own `hllsetLUT` entry (`th > 0`);
  * the eLLM discovers its `H(t)` and broadcasts it;
  * its own door guard answers `can_restore = true` (coverage = 1.0);
  * the agent selects **self**, and the eLLM generates.
* **Growth by attachment.** Attaching a new LLM means standing up its own
  ewm-sm side-car (own LUTs, own `hllsetLUT`, own `H(t)`, own codebook gate,
  own `door.py`). It starts answering broadcasts automatically — the eLLM now
  receives two door verdicts instead of one. **Nothing in the eLLM changes**:
  no registry update, no retraining, no architecture switch. This is the
  host-agnosticism experiment of phase 6, run live.

The development order follows this shape: **phases 0–5 build and validate the
whole loop at N=1** (the eLLM's own door is the only responder); **phase 6
attaches a second LLM** and shows the broadcast collects two verdicts with no
change to the eLLM.

---

## 3. Layered architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 · SERVING & TRAINING (Python, PyTorch · ewm-nanolm env)        │
│  ┌────────────────────────────┐        ┌────────────────────────────┐   │
│  │  SERVER (broadcast+gen)    │        │  TRAINER (policy learner)  │   │
│  │  · reads committed policy  │        │  · owns scheduler          │   │
│  │  · never blocks            │        │  · WIP → learn → guard     │   │
│  │  · broadcasts / generates  │        │  · commits at slow time    │   │
│  └─────────────┬──────────────┘        └─────────────┬──────────────┘   │
│                │  read tip                           │  commit          │
│                ▼                                     ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  THE STORE — ewm-state-machine three-location memory              │  │
│  │    WIP (in-memory) · cache (Arrow, c:<sha1>) · persisted (ewm-git)│  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 · eLLM RUNTIME (Python, PyTorch · RTX 3060)                    │
│  ellm.model.base        Qwen2.5-1.5B-Instruct (local_files_only)        │
│  ellm.model.ranker      broadcast policy: discover/order HLLSet         |
|                         candidates                                      │
│  ellm.model.lora        PEFT/LoRA adapters (attention projections)      │
│  ellm.travel            the travel-loop controller (six-step turn,      │
│                         shrink/expand, surprise-gated door policy)      │
│  ellm.curriculum        (state, options, decision, confidence) samples  │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 · LATTICE PROTOCOL (JSON lines over stdin/stdout)              │
│  ellm.lattice           client; commands: ping/ingest/materialize/      │
│                         state/commit/hllset_lut/gate/to_prompt/         │
│                         door_probe/policy_put/policy_get/setkey/        │
│                         setkey_hex/popcount_hex                         │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 1 · LATTICE SIDE-CAR (Rust workspace: lattice/)                  │
│  ellm-lattice           CLI service reusing ewm-state-machine crates    │
│  ellm-attn              pure-software attention engine (no FPGA):       │
│                         KStorage, BitKeyTable, KBridge, setkey, hybrid, │
│                         context_vocab, token_mask, moe                  │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 0 · EWM STATE MACHINE (Rust, reused by path, unchanged)          │
│  hllset-contracts · hllset-cid · hllset-core · hllset-lut ·             │
│  hllset-morphisms · hllset-storage · context-tree · ewm-git · ewm-app · │
│  ewm-boolring · ewm-scene · ewm-ops · ewm-cache                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Layer 0 — EWM State Machine (reused, never modified)

The eLLM consumes these existing pieces directly:

| Piece | Role in eLLM |
| --- | --- |
| `hllset-core::HLLSet` | the node / waypoint (Roaring bitmap over the 32 768-bit plane) |
| `hllset-morphisms::{ingest, materialize}` | the door (padded 1/2/3-gram channels; LUT-first, ordered De Bruijn restoration) |
| `hllset-morphisms::gate` | per-host codebook gates — host encodings as languages |
| `HllsetLut` (`hllset-morphisms` / `ewm-git`) | per-ewm-sm registry of generated HLLSets (`<name, SHA1, TH>` / `<ObjectId, TH>`) — the candidate **gate** and selection **confirmation** (§2.3) |
| `ewm-git::{Repository, LatticeState, ObjectId}` | commit DAG, tip, `D/R/N`, recovery; **the KV cache is the commit DAG** |
| `ewm-app::{StateMachine, StateCache, CodebookEncoder}` | the [UM] turn loop; `S(t)`/`H(t-1)` outside the [UM] |
| `ewm-boolring::{BoolWindow, BoolBasis, RingStats}` | ring basis = distinctive oppositions; soft/hard key, residual, rotation |
| `ewm-scene::{noether, sidecar, project, subframes}` | trajectory readings: BSS series, `D/R/N`, frame projections |
| `context-tree::ContextTree` | Merkle presentation of the working set |
| `ewm-cache` | Arrow IPC spills for derived cache objects (optional, phase 4+) |

### 3.2 Layer 1 — Lattice side-car (Rust)

A thin Rust workspace (`lattice/`) with two crates:

* **`ellm-lattice`** — a JSON-lines daemon/CLI that wraps the layer-0 APIs
  behind one protocol. It is the only Rust-facing module on the Python side
  (mirrors the existing `ewm-state-machine/trainer/ewm.py` pattern). Commands:

  ```text
  ingest    {turn_id, tokens[]}                   → {key, g1, popcount, coverage}
  state     {}                                    → {tip, S(t), H(t-1), D, R, N,
                                                    ring:{dim,residual,in_span,
                                                    soft_key,hard_key,spill},
                                                    windows, coverage}
  materialize {node, beam?, no_order?}            → {tokens[], coverage, order}
  commit    {message}                             → {tip}
  noether   {frames}                              → D/R/N series
  sidecar   {frames}                              → BSS trajectory, ring frame
  setkey    {query_node, context_nodes[]}         → {scores[], candidates[][]}
  hybrid    {dense[], lattice[], lambda}          → {scores[]}
  kstorage  {tokens[]}                            → {keys[], candidates[], confidence}
  gate      put|list {user, codebook[]}           → host vocabulary gates
  ```

* **`ellm-attn`** — the exact engine of
  `attention_lattice_traversal.tex`, implemented as **pure Rust software** (the
  FPGA line is out of scope for now). It follows the cortex design notes
  (`HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md`, `HLLSET_K_SPACE_MATH.md`) and the
  paper's implementation map, with no FPGA dependencies. Modules: `kstorage`
  (tier 1 `TokenLutStorage`, tier 2 `CatalogLutStorage` quorum), `bit_table`
  (`BitKeyTable` 32 768 × d, Mode A), `bridge` (`KBridge`), `setkey` (Mode B:
  overlap / Jaccard / BSS), `hybrid` (Mode C: `dense + λ·lattice`),
  `context_vocab`, `token_mask`, `moe` (Expert Think Tank).

Dependencies are **by path** into `../ewm-state-machine/crates/*` — never
vendored, never re-declared (GOVERNANCE.md §5).

### 3.3 Layer 2 — Lattice protocol

Python talks to Rust over **JSON lines on stdin/stdout** (a subprocess), the
same seam the state machine's `trainer/ewm.py` already uses. Phase 4 may add an
Arrow IPC path via `ewm-cache` for bulk trajectory transfers; the protocol stays
the stable interface either way.

### 3.4 Layer 3 — eLLM runtime (Python, PyTorch)

* **`ellm.model.base`** — loads `Qwen/Qwen2.5-1.5B-Instruct` from the local HF
  cache (`local_files_only=True`), fp16/bf16, on the RTX 3060
  (`CUDA_VISIBLE_DEVICES=1`: the Quadro M1200 is CUDA0 on this laptop).
* **`ellm.model.ranker`** — the **broadcast policy** (discovery, not
  prediction). Input: the query HLLSet `Q = I(query)`, the candidate HLLSets
  (per-host `H(t)` nodes and sub-nodes from the commit DAG), and the lattice
  state vector (BSS, `D/R/N`, ring soft/hard key, residual, spill, windows).
  Output: **which HLLSet candidates to broadcast, in which order**. It never
  names hosts and never decides the winner — the doors do (§2.3).
* **`ellm.model.lora`** — PEFT/LoRA adapters on the attention projections (and
  the broadcast-policy head). The first learned policy is a LoRA student
  distilled from the **native ring-basis labels** — which HLLSets are worth
  broadcasting first (soft-label KL) — validated against `routing_log.jsonl`
  from notebooks 13–21.
* **`ellm.travel`** — the travel-loop controller. One turn is the state
  machine's six-step transformation — **encode, ingest, join, ring-push,
  commit, materialize** — plus the door policy:
  * **shrink**: meet context with query, restrict to a sub-lattice, minimal cover;
  * **expand**: join neighbors, lower BSS threshold, add bootstrap n-gram views;
  * **surprise gate** (paper open question 1, first design): open the door when
    novelty `N` or ring residual spikes; otherwise keep the compressed node.
* **`ellm.curriculum`** — samples `(state, options, decision, confidence)` from
  the lattice trajectory and the routing logs. Strategies: `RecentFirst`,
  `NoveltyWeighted`, `ConfidenceWeighted`, `BalancedDnr`.

### 3.5 Layer 4 — Serving and training (two processes, one store)

Absorbed from `emerging-llm/docs/ARCHITECTURE.md` §2 and §9, unchanged in
spirit:

* **Server process** — broadcast dispatcher + generator + selection agent.
  Reads the committed policy tip, serves requests, never blocks, never
  schedules. Refreshes the policy between requests (hot-swap).
* **Trainer process** — owns the scheduler, samples the curriculum, learns WIP,
  runs the guard set, and **commits at slow time**. Publication is the commit:
  the server sees a policy only when it is committed.
* The only channel between them is the **store**: WIP (in-memory) → Arrow cache
  → persisted `ewm-git` tip. IICA makes torn states unrepresentable; recovery is
  a read of the tip.

---

## 4. One turn, end to end

```text
 1  encode      user prompt → token ids (base model tokenizer)
 2  ingest      μ(tokens) → query HLLSet Q = I(prompt)          [Rust: ingest]
 3  traverse    score context nodes vs Q (BSS/Jaccard/overlap), [Rust: setkey]
                select working set (threshold θ); shrink/expand
 4  materialize selected node → token context C_tok             [Rust: materialize]
                (coverage guard: confidence = 1.0)
 5  broadcast   gate candidates via per-host hllsetLUTs;        [Rust: setkey
                discover which HLLSets to send; broadcast to       + Python: policy]
                every door; collect verdicts {coverage, 
                can_restore}; agent selects target (0→fallback, 
                1→route, >1→user)
 6  move        chosen speaker generates inside C_tok            [Python: base LLM]
                (self = Mode C hybrid attention, λ optional)
 7  ingest      generated tokens → new node N'                   [Rust: ingest]
 8  commit      advance tip; D/R/N + ring-push                   [Rust: commit]
 9  next turn   tip = H(t-1); commit DAG is the KV cache
```

Step 5 is the *actualization* of `langue_as_a_lattice.tex` — discovery,
broadcast, door verdicts, and selection; steps 2–4 and 7–8
are the *door*; steps 3 and 6 are the *map lookup* and the *locomotion* of
`attention_lattice_traversal.tex`.

---

## 5. The exact engine: Mode A / B / C in eLLM

| Mode | Paper definition | eLLM use |
| --- | --- | --- |
| **A** address-key | key = learned bit cell `E_bit[bit(t)]`; 32 768 × d table | optional: lattice-shaped key partition for the base model's K side (phase 5+) |
| **B** set-key | `Q_hll = I(prefix)`, `K_hll = C`, score = BSS/Jaccard/containment, value = `M(K_hll ∩ Q_hll)` | **candidate discovery** for the broadcast (step 3); the exact routing decision is the doors' coverage (step 5) |
| **C** hybrid | `q·k/√d + λ·φ(Q_hll, K_hll)` | the base LLM's attention with a lattice steering term; λ learned or scheduled |

The three modes are **one temperature axis** (softmax is the entropic
relaxation of the threshold). Development order: **B first** (exact, no
gradients — proves the door and the curriculum), **C second** (λ as a learned
LoRA parameter), **A last** (only if the learned bit table proves useful on top
of the pretrained model).

---

## 6. What gets learned, against what loss

1. **Broadcast-policy accuracy (the routing objective, reformulated).**
   Discover/order the HLLSet candidates so that the **native ring-basis
   label** — the HLLSet whose coverage of the materialized context is highest
   (BSS, soft key `w`, hard key `c`, spill `s` over the evolving RREF basis) —
   is broadcast first. The final routing decision is then the doors' exact
   coverage verdicts; `routing_log.jsonl` (Jev / Laya / structural-Qwen
   decisions) is validation only.
2. **Structural fitness (EWM objective).** The selected door's coverage of
   the broadcast HLLSet must be 1.0 (`|tokens \ codebook_gate(chosen door)|
   → 0`). Learning and the invariant are the same thing.
3. **Conservation (Noether objective).** Prefer updates that rearrange the
   encoding registry / broadcast order over accumulating capacity; the
   gradient-free `rank-lattice` path follows the LoRA path.

**First learned policy (decided):** LoRA/PEFT on Qwen2.5-1.5B-Instruct — the
broadcast-policy head distilled from native ring-basis labels, validated
against `routing_log`, with the door coverage guard as the first defense.

---

## 7. Guards — the acceptance contract

Ported from `emerging-llm` §7, extended with lattice-specific guards:

| Guard | Vetoes when |
| --- | --- |
| `NonEmptyGuard` | the curriculum produced no broadcast samples |
| `RegressionGuard` | probe broadcast-policy accuracy dropped beyond slack |
| `RegistrationGuard` | a broadcast HLLSet is not registered (`th > 0`) in any host's `hllsetLUT` |
| `CoverageGuard` | no door accepted the broadcast HLLSet (no verdict has coverage = 1.0) |
| `ConfirmationGuard` | the chosen door's optional ring/trajectory confirmation fails (residual outside its ring span, `D/R/N` inconsistent with its tip, quorum unmet) |
| `ProgressGuard` | relative probe gain below `accept_min_gain` |
| `RoundTripGuard` | door round-trip coverage ≠ 1.0 before materialize-to-prompt |
| `SurpriseGate` | (policy) door opens without novelty/residual signal — protects the compressed context |
| `SeparationGuard` | two candidate `H(t)` are contextually too close (BSS above threshold) — collapses the ranking |

Runtime confidence gating is unchanged from the router line: a top-1
confidence below `confidence_floor` falls back to a safe host (or to self).

---

## 8. Repository layout

```text
ellm/
├── docs/
│   └── ARCHITECTURE.md          # this document
├── lattice/                     # Rust side-car workspace (by-path deps only)
│   ├── Cargo.toml               # workspace: ellm-lattice + ellm-attn
│   └── crates/
│       ├── ellm-lattice/        # JSON-lines service over ewm-state-machine
│       └── ellm-attn/           # pure-software exact engine (Mode A/B/C, K-storage)
├── ellm/                        # Python package (ewm-nanolm env)
│   ├── lattice/                 # Rust side-car client (subprocess, JSON lines)
│   ├── model/                   # base model, LoRA adapters, broadcast policy
│   ├── travel/                  # travel-loop controller + door policy
│   ├── curriculum/              # sampling strategies
│   ├── train/                   # window driver, guards, checkpoints
│   └── serve/                   # broadcast + generator + selection + HTTP demo server
├── deploy/                      # podman-compose demo (Containerfile, build.sh, run.sh,
│                                #   podman-compose.yml, podman-compose.gpu.yml)
├── notebooks/                   # one notebook per phase (00–06)
├── scripts/                     # smoke tests, ingestion, growth experiment
├── tests/                       # Python + Rust integration tests
├── pyproject.toml
└── README.md
```

Dependency direction is the permission system: Python depends on the Rust
side-car only through the JSON-lines protocol; the Rust crates depend only on
`ewm-state-machine` crates by path; nothing mutates the lattice; nothing
re-declares a soldered invariant.

---

## 9. GPU budget — RTX 3060 12 GB

| Component | Memory (approx.) |
| --- | --- |
| Qwen2.5-1.5B-Instruct, fp16 weights | ~3.1 GB |
| KV cache (≤ 2048 tokens, fp16) | ~1–2 GB |
| LoRA adapters (r=16–64 on attention) | tens of MB |
| Training: adapter grads + optimizer (AdamW) | ~1–2 GB |
| **Total inference / training headroom** | **well under 12 GB** |

* Default base model: **Qwen2.5-1.5B-Instruct** (cached). Qwen2.5-3B-Instruct
  (~6 GB fp16) is the stretch option if the 1.5B head underfits.
* The lattice runs **on CPU**: HLLSets are 4 KB Roaring bitmaps, set ops are
  bitwise, and the `BitKeyTable` (32 768 × 1536 fp32 ≈ 200 MB) is small.
* `CUDA_VISIBLE_DEVICES=1` selects the RTX 3060 (Quadro M1200 is CUDA0), per
  the existing notebook convention.
* Training with LoRA (not full fine-tune) is a hard budget decision, not a
  preference: full fine-tune of 1.5B would need ~4× weights for
  grads+moments+variance, which still fits but leaves no serving room. LoRA
  keeps serving and training on the same GPU.

---

## 10. Implementation phases

Phases 0–5 run at **N=1** (the eLLM is the only LLM in the network and the
only candidate); phase 6 attaches a second LLM and demonstrates growth.

| Phase | Deliverable | Status |
| --- | --- | --- |
| **0** | repo scaffold: this doc, `lattice/` builds, Python package skeleton, GPU smoke test, JSON-lines protocol v0 | **done** |
| **1** | wire the real lattice: `ellm-lattice` service over `ewm-state-machine` (ping/ingest/materialize/state/commit/hllset_lut/setkey); `ellm-attn` complete (kstorage, bit_table, bridge, setkey, hybrid, token_mask, context_vocab, moe); routing-log ingestion script + `LatticeState` snapshots | **done** |
| **2** | the door: materialize-to-prompt with coverage invariant; the eLLM's own codebook gate (single-LLM case); probes held out | **done** |
| **3** | door-guard broadcast seam at N=1 (degenerate self-selection): Mode B discovery (`setkey`/`setkey_hex`) + `door_probe` verdict on the eLLM's own side-car; `BroadcastDispatcher` + `NativeStubPolicy` + selection (0→fallback, 1→hire, >1→user) | **done** |
| **4** | two-process split + closed travel loop at N=1: `TravelServer` (broadcast + generate + selection, policy hot-swap) and `Trainer` (scheduler + WIP learner + guard set, commit-as-publication via the `@policy` slot in the shared ewm-git store); `Service<LooseStore>` (disk-persisted side-car, `ellm-lattice <repo>`); surprise-gated door policy v0 | **done** |
| **5** | first learned policy at N=1: `QwenBroadcastRanker` (LoRA on Qwen2.5-1.5B + score head over lattice-state prompts), soft-label KL vs native ring-basis labels (`collect_samples`), real guard set (probe accuracy, regression slack, progress gain), `QwenGenerator` behind the `Generator` protocol; evaluation on held-out probes | **done** |
| **6** | growth experiment: **attach a second LLM** — `HostLLM` (own ewm-sm + `hllsetLUT` + `H(t)` + `door.py`), `DoorNetwork` (one side-car per LLM, attach = add an endpoint), `turn_hex` (per-turn HLLSet as the broadcast candidate); disjoint codebooks route to self/host/fallback, overlapping codebooks escalate to the CEO; **no change to the eLLM** (`scripts/growth_experiment.py` publishes the table) | **done** |
| **7** | *(candidate)* **the NCM bridge**: HLLSet lattice as the **coarse level** of coarse-to-fine generation (Next-Concept Modeling, HazyResearch 2026). The sub-lattice is the scale pyramid (shrink = coarser, expand = finer); the door is the concept→detail decoder; a parallel fine decoder inside the materialized node is the NCM-style speed engine. See §11.7 | candidate |

Each phase lands with a notebook in `notebooks/` and a baseline test count, in
the gen2 style.

---

## 11. Open questions and risks

1. **Surprise-gated door policy** (attention paper, open question 1): which
   traversal signal (novelty `N`, ring residual, `D/R/N` crossover) triggers
   materialization? Phase 4 v0 uses residual spike; the rest is measured.
2. **Soft support with hard bits** (open question 2): can the exact engine's
   threshold be differentiable without losing the bitmap? This decides whether
   Mode C is a bias term or a true differentiable traversal.
3. **Canonical waypoint selection** (open question 3): a tiling policy over the
   commit DAG analogous to FlashAttention's traversal order. Phase 2 ships
   BSS-threshold selection; the policy is refined in phase 5.
4. **Candidate separation**: candidate `H(t)` nodes must be *contextually
   different* (BSS separation); otherwise they collapse to the same lattice
   region and the ranking is degenerate.
5. **Determinism/reproducibility**: fixed seeds, content-addressed checkpoints,
   `local_files_only=True`; every experiment must replay bit-identically.
6. **Cost sharing**: policy learning competes with serving for the same RTX
   3060; the postponed-commit scheduler (§3.5) is the throttle.
7. **The NCM bridge (coarse-to-fine generation).** Next-Concept Modeling
   (HazyResearch, 2026) generates text from a hierarchy of learned discrete
   codes — few coarse codes first, then parallel refinement per level
   (9–31× faster than diffusion). The eLLM already runs the same shape with a
   **structural** coarse level: the HLLSet lattice node is the concept, the
   door (`materialize`, coverage = 1.0) is the concept→detail decoder, and
   attention inside the materialized node is the fine generation. Candidate
   direction: the **sub-lattice as the scale pyramid** (shrink = coarser,
   expand = finer — no learned VQ hierarchy needed), the broadcast as
   parallel next-concept selection, and an NCM-style parallel fine decoder
   inside the node.

   **Order is already structural, not an open risk.** For sequential streams
   (text, image patches) the ordered path uses the **padded n-gram scheme**:
   ingest writes 1-/2-/3-gram channels, and materialize restores order via
   the **De Bruijn chain** anchored at the start pad — count-constrained
   TF-scored decoding over the graph (1-gram = vocabulary, 2-/3-gram =
   edges), with beam 1 = greedy, beam > 1 = top-k paths. The native LLM
   encodings are therefore delivered **in restored order**; the flat n-seed
   scheme (unordered, per-token atoms) is reserved for codebook gates and
   catalog keys, where order is intentionally irrelevant. Open sub-questions
   become: (a) does the lossy HLLSet sketch carry enough "gist" at the
   coarsest levels? (b) can the De Bruijn restoration double as NCM's
   "cumulative reconstruction" conditioning for the fine decoder (its beam
   decoding is already the same shape as LLM token generation)? (c) does Mode
   C hybrid scoring give the binary lattice the smoothness the fine decoder
   needs? Risk: the lattice makes the coarse level exact, not faster — the
   speed gain must come from the parallel fine decoder.

---

## 12. Invariants (from GOVERNANCE and the papers)

1. **Two processes, one channel.** Server and trainer meet only at a commit.
2. **Training never mutates the lattice.** The corpus is read-only; a window
   produces an update and a checkpoint; a veto rolls back.
3. **Publication is atomic and lock-free.** IICA makes torn states
   unrepresentable; the server reads whatever tip is current.
4. **The door is faithful, not lossless.** Round-trip recovers the 1-HLLSet
   exactly and structural layers up to collision ambiguity; exact prose is kept
   as an original node, not a derived one.
5. **Coverage is the exactness gauge.** `LUT.bits ⊇ bits(C)`; materialize
   without coverage = 1.0 is refused (RoundTripGuard).
6. **No tokenizer training.** Host encodings (`tid{n}` / 4-byte LE) are inputs,
   never parameters.
7. **Host-LLM agnostic by construction.** The eLLM never names hosts: it
   discovers HLLSets and asks every door; the doors answer with exact coverage
   verdicts. Adding a host = its side-car starts answering broadcasts; the
   eLLM is unchanged and the lattice never changes.

---

## Appendix A — concept-to-code map

| Paper concept | Artifact |
| --- | --- |
| bit plane, HLLSet | `hllset-core::HLLSet` (Roaring bitmap) |
| geometry `P, M, b` | `hllset-contracts` (`P`, `M`, `BITS_PER_REG`) |
| morphism `μ` / the door | `hllset-morphisms::{ingest, materialize}` |
| working state `S(t)`, base `H(t-1)` | `ewm-app::StateCache::{working, tip}` |
| `D/R/N` | `ewm-git::LatticeState` + `ewm-scene::noether` |
| ring basis, push, rotation | `ewm-boolring::{BoolWindow, RingStats}` |
| commit DAG, tip, KV cache | `ewm-git::{Repository, ObjectId}` |
| trajectory frame, soft key | `ewm-scene::{sidecar, project, subframes}` |
| world, bridge | hash configuration `(h, s)`; `ewm-scene` gates |
| `hllsetLUT` (gate + confirmation) | `hllset-morphisms::HllsetLut` (`<name, SHA1, TH>`) / `ewm-git::HllsetLut` (`<ObjectId, TH>`) |
| context shrink / expand | sub-lattice restriction, cover, union, threshold |
| K-storage interface | `ellm-attn::kstorage::KStorage` |
| Mode A key table | `ellm-attn::bit_table::BitKeyTable` |
| Mode B/C scores | `ellm-attn::{setkey, hybrid}` |
| dispatcher `discover → broadcast → door verdicts → select` | `ellm.model.ranker` (broadcast policy, LoRA) + every side-car's door + agent selection |
| travel loop | `ellm.travel` (six-step turn controller) |
| curriculum `(state, options, decision, confidence)` | `ellm.curriculum` |

---

## Appendix B — glossary

| Term | Meaning |
| --- | --- |
| **eLLM** | the Emerging LLM: a local LLM with lattice memory, a door-guard broadcast dispatcher, and the closed travel loop |
| **map** | the EWM HLLSet lattice — exact, persistent, content-addressed |
| **door** | materialize / ingest — HLLSet context ⇄ token context |
| **move** | attention inside a materialized node (locomotion) |
| **actualization** | the broadcast of discovered HLLSet candidates + each door's coverage verdict + the agent's selection (user breaks ties) |
| **host LLM** | an existing LLM (self, Qwen2.5-1.5B, Bonsai, DeepSeek-coder, …) whose encoding restores real tokens |
| **hllsetLUT** | the per-ewm-sm append-only, touch-counted registry of all HLLSets that ewm-sm generated — used as the candidate **gate** and the selection **confirmation** |
| **top HLLSet** | the candidate set: every host's `H(t)` nodes in the lattice — discovered and broadcast, not registered |
| **door verdict** | one door guard's answer to a broadcast: `{coverage, can_restore}` against its own codebook gate |
| **LatticeState** | `{S(t), H(t-1), D, R, N, ring, windows, coverage}` |
| **native ring-basis label** | the EWM-native training label: soft/hard key + spill over the RREF basis |
| **curriculum** | ranked `(state, options, decision, confidence)` samples |
| **guard** | acceptance condition; any veto rolls the window back |
| **checkpoint** | a content-addressed node recording an accepted policy window |
| **slow time** | the trainer's postponed-commit moment (low traffic) |

---

## Appendix C — `ellm-attn` implementation detail

`ellm-attn` is the **attention-support engine**: it computes the K side and the
V side of attention **on `tid{n}` ids — the side-car's hashable inscription
of host-LLM encodings**, never on dense embeddings. The dense machinery
(Q/K/V projections, softmax, multi-head concatenation, FFN) stays in PyTorch.
The division of labour is exactly the
support–union lemma: `ellm-attn` decides **which bits/keys are in the working
set** (the support); PyTorch decides **the convex coordinates inside the
support** (the weights).

### C.1 Input contract

```text
host LLM encodings (f32 vectors)
        │  ewm-app::CodebookEncoder::encode   (cosine nearest-anchor quantization)
        ▼
tid{n} ids  — the side-car's inscription (TokenId = u32 → `tid{n}` bytes via
        │     token_in_bytes; a compact 4-byte LE variant exists via token_in_bytes_le)
        │  ellm-attn key_of(tid_bytes)
        ▼
bit cells / HLLSets  (Roaring bitmaps over the 32 768-bit plane)
```

`tid{n}` is the only encoding that crosses into the lattice, because the
geometry is soldered (`P=10`, `M=1024`, `b=32`); everything else — raw
embeddings, arbitrary token strings — must first be inscribed as `tid{n}`.
Host encodings and their EWM hash values are **isomorphic up to hash
collision**: algebraically equal for the lattice, with the hash the
technically consistent form (fixed-size, deterministic, host-agnostic).

`ellm-attn` uses the soldered geometry from `hllset-contracts::hashing`:
`P=10`, `M=1024`, `BITS_PER_REG=32`, `TOTAL_BITS=32 768`,
`token_to_position(bytes) → (reg, tz)`, `bit = reg*32 + tz`,
`token_to_position_seeded(bytes, seed)`, `hash_to_position(hash)`,
`murmur3_hash` / `murmur3_hash_seeded`.

### C.2 Modules

| Module | Implements | Primary APIs |
| --- | --- | --- |
| `kstorage` | K-storage: token → key, HLLSet → candidates, coverage | `KStorage` trait; `TokenLutStorage` (tier 1); `CatalogLutStorage` (tier 2) |
| `bit_table` | Mode A learned key table | `BitKeyTable` (32 768 × d row-major f32) |
| `bridge` | token → key vector routing | `KBridge::key_vector` |
| `setkey` | Mode B lattice scores | `overlap`, `jaccard`, `bss_coverage`, `bss_symmetric`, `retrieve`, `score_many` |
| `hybrid` | Mode C combination | `hybrid_score`, `hybrid_logits` |
| `context_vocab` | incremental context vocabulary under D/R/N | `ContextVocabulary::step(D, N)` |
| `token_mask` | content-addressed gate (murmur3 seed-0 hashes) | `TokenMask` |
| `moe` | Expert Think Tank waypoint ranking | `Observation`, `Expert`, `RankedExpert`, `MoE::rank` |

#### C.2.1 `kstorage`

```rust
pub trait KStorage {
    fn key_of(&self, token: &[u8]) -> KeyRef;
    fn candidates(&self, hllset: &HLLSet) -> Vec<Vec<u8>>;
    fn confidence(&self, hllset: &HLLSet) -> f64;   // resolved bits / total bits
    fn register(&mut self, token: Vec<u8>);          // append-only, idempotent
    fn resolve(&self, hash: u64) -> Option<Vec<u8>>;
}
```

* **`TokenLutStorage` (tier 1)** — wraps `hllset-lut::LutIndex`.
  `key_of` = `token_to_position(bytes)` → `KeyRef::Cell(reg*32+tz)`.
  `candidates` = `lut.materialize(hllset)` (LUT-first, keeps every collided
  candidate). `confidence` = fraction of set bits whose fiber is non-empty.
* **`CatalogLutStorage` (tier 2)** — local per-seed catalog, seeds `[0,1,2]`,
  quorum 2-of-3. `key_of` → `KeyRef::Cells([c0,c1,c2], 2)`. `candidates`:
  a value passes iff ≥ quorum of its seed cells are set in the HLLSet.
* Additions over the old reference: `from_tids(Vec<TokenId>)` constructors
  using `token_in_bytes_le`, and `key_of_many(&[TokenId]) -> Vec<KeyRef>` for
  batch serving.

#### C.2.2 `bit_table` (Mode A)

`BitKeyTable { d: usize, data: Vec<f32> }`, row-major, 32 768 rows. Same shape
as the reference (`new`, `from_fn`, `get`, `get_mut`, `set`, `rows`), plus:

* `to_bytes()/from_bytes()` — little-endian f32 flat dump;
* `save(path)/load(path)` — the exchange format for the Python side.

Size at `d=1536`: 32 768 × 1536 × 4 B ≈ 201 MB (CPU-resident; Python may keep
the trainable copy as `torch.nn.Embedding(32768, d)` and sync down fp32).

#### C.2.3 `bridge`

`KBridge<S: KStorage>`: `key_vector(token, &BitKeyTable) -> Vec<f32>` —
tier 1 returns the single cell row; tier 2 returns the centroid of the
multi-seed cell rows. `key_vector_with` adds a residual in place.

#### C.2.4 `setkey` (Mode B)

Exact bit-level scores, no dot products:

```text
overlap(a,b)     = |a ∩ b|                        (exact popcount)
jaccard(a,b)     = |a ∩ b| / |a ∪ b|
bss_coverage(c,q)= |c ∩ q| / |q|                  (directed: context covers query)
bss_symmetric(a,b) = min(τ(a,b), τ(b,a))
retrieve(c, q)   = (bss_coverage(c,q), candidates from c ∩ q)
score_many(q, [c_i]) -> Vec<f64>                  (batch, hot path)
```

`score_many` is the workhorse for candidate discovery (one query vs the
commit DAG nodes) ahead of the broadcast.

#### C.2.5 `hybrid` (Mode C)

```text
hybrid_score(dense, lattice, λ) = dense + λ · lattice
hybrid_logits(dense[], lattice[], λ)
```

The lattice term comes from Rust, the dense logits from PyTorch; they are
combined **before softmax** (truncated softmax at threshold θ is the hard
traversal, per the correspondence theorem).

#### C.2.6 `context_vocab` + `token_mask`

Incremental vocabulary under the `D/R/N` churn:

```text
V(t) = ( V(t-1) \ M(D(t)) ) ∪ M(N(t))
```

The vocabulary is a `TokenMask` of content addresses (murmur3 seed-0 hashes),
so the mask can never drift from the LUT; departure is a lattice operation,
never a LUT deletion (the LUT stays append-only).

#### C.2.7 `moe` (Expert Think Tank, later)

Content-address shuffle (`sort_by_sha1`) over recent observations →
convolution frames `F_k` → experts `E_k = ∪F_k` → ranked by
`bss_coverage(E_k, R(t))` against the recent observation → top-k + threshold
`τ_min` → the focused working set. This is the candidate source for the
waypoint-selection policy (paper open question 3), not a phase-1 dependency.

### C.3 Mode C inside the PyTorch model

The paper's Mode C is `σ(i,j) = q_i·k_j/√d + λ·φ(Q_hll, K_hll)`. Concretely
in the eLLM:

1. **Rust precomputes, once per turn**, a per-position HLLSet `X_i` for each
   token position — the token's 1-/2-/3-gram atom (≤ ~9 bits). These are
   tiny Roaring bitmaps; the per-position φ is a 9-bit AND.
2. **Rust returns the lattice matrix** `L[i][j] = φ(X_i, X_j)` for the current
   window (e.g. `bss_coverage` or `overlap`), or computes it on demand per
   layer. n=2048 → 4 M tiny bit ops, same order as attention but far cheaper
   per op than a d-dim dot product.
3. **Python adds** `λ · L` to the attention logits before softmax.
4. **λ** is a per-layer scalar, LoRA-trained or scheduled (phase 5); λ=0
   reproduces the pretrained baseline exactly.

The **prefix/node-level** HLLSets (`Q_hll = I(prefix)`, node sketches) are
*not* used per position — they are the Mode B waypoint-selection objects. This
keeps Mode C O(n²)-cheap while Mode B stays exact and node-scoped.

### C.4 Protocol commands (Layer 2)

```text
attn_key_of     {tids[]}                 → {keys[], quorum?}
attn_candidates {hllset}                 → {candidates[], confidence}
attn_setkey     {query, nodes[]}         → {scores[]}
attn_hybrid     {dense[], lattice[], λ}  → {scores[]}
attn_bit_table  get|set|load|save        → BitKeyTable exchange
attn_vocab      {D, N}                   → VocabDelta
```

### C.5 Testing strategy

* Port the reference invariants: `key_of` matches hash decomposition;
  `candidates` recover all tokens; `confidence == 1.0` iff the LUT covers the
  sub-lattice; catalog quorum is 2-of-3; `set`/`get` roundtrip; hybrid
  length-mismatch panic.
* `tid{n}` roundtrip: `token_in_bytes_le ↔ parse_token_id_le`.
* `score_many` consistency with scalar `bss_coverage`/`overlap`.
* `BitKeyTable` `to_bytes/from_bytes` roundtrip; cross-check against a
  `torch.nn.Embedding(32768, d)` tensor loaded in Python.
* Mode C numeric check: hand-computed φ on toy bit sets vs Rust output.
* Coverage invariant on real `CodebookEncoder` tids from the side-car loop.

### C.6 Deltas from the parked `hllset-attn` reference

1. Target the **current `ewm-state-machine` APIs** (hashing re-exports,
   `HLLSet` methods, `LutIndex`).
2. **Serialization** for `BitKeyTable` (Python exchange).
3. **Batch APIs** (`key_of_many`, `score_many`) for serving latency.
4. **`tid{n}` first-class**: constructors and protocol commands take
   `TokenId`/4-byte LE, matching the side-car loop.
