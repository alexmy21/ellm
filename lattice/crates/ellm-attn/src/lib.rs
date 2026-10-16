//! # ellm-attn — the exact engine of `attention_lattice_traversal.tex`
//!
//! The **attention-support engine**: it computes the K side and the V side of
//! attention **on `tid{n}` ids — the side-car's hashable inscription of host
//! LLM encodings**, never on dense embeddings. `tid{n}` is the only encoding
//! compatible with the soldered EWM lattice geometry (`P = 10`, `b = 32`):
//! the morphism hashes every token (murmur3, per-world seed) into a bit
//! address on the shared 32 768-bit plane. Host encodings and their EWM hash
//! values are isomorphic up to hash collision — algebraically equal for the
//! lattice, with the hash the technically consistent form: fixed-size,
//! deterministic, host-agnostic. The dense machinery (Q/K/V projections,
//! softmax, multi-head concatenation, FFN) stays in PyTorch. The division of
//! labour is the support–union lemma: `ellm-attn` decides **which bits/keys
//! are in the working set** (the support); PyTorch decides **the convex
//! coordinates inside the support** (the weights).
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`kstorage`] | `KStorage` trait + tier-1/2 LUT storages (token → key, HLLSet → candidates) |
//! | [`bit_table`] | `BitKeyTable` — the learned 32 768 × d key table (Mode A) |
//! | [`bridge`] | `KBridge` — routes a token through K-storage into a key vector |
//! | [`setkey`] | HLLSet-native attention scores: overlap, Jaccard, BSS (Mode B) |
//! | [`hybrid`] | hybrid dense + lattice score (Mode C) |
//! | [`token_mask`] | content-addressed token gate over the LUT |
//! | [`context_vocab`] | incremental context vocabulary under `D/R/N` churn |
//! | [`moe`] | Expert Think Tank — content-addressed expert ranking |

pub mod bit_table;
pub mod bridge;
pub mod context_vocab;
pub mod hybrid;
pub mod kstorage;
pub mod moe;
pub mod setkey;
pub mod token_mask;

pub use bit_table::BitKeyTable;
pub use bridge::KBridge;
pub use context_vocab::{ContextVocabulary, VocabDelta};
pub use hybrid::{hybrid_logits, hybrid_score, HybridConfig};
pub use kstorage::{
    CatalogLutStorage, KStorage, KeyRef, TokenLutStorage, DEFAULT_CATALOG_SEEDS,
};
pub use moe::{
    build_experts, focus, frames, rank, shuffle_by_sha1, Expert, Observation, RankedExpert,
};
pub use setkey::{
    bss_coverage, bss_symmetric, jaccard, overlap, retrieve, score_many, Retrieval,
};
pub use token_mask::TokenMask;
