//! Incremental context vocabulary maintenance.
//!
//! The context sub-lattice evolves through commits with `D/R/N`. Only the
//! new bits carry unknown vocabulary, so instead of re-materializing the
//! whole context union, the token vocabulary is maintained incrementally:
//!
//! ```text
//! V(t) = ( V(t-1) \ M(D(t)) )  ∪  M(N(t))
//! ```
//!
//! where `M = KStorage::candidates`. Per-step work is proportional to the
//! churn `|N ∪ D|`, not to `|C(t)|` — the natural bound on context
//! vocabulary.
//!
//! The vocabulary is a [`TokenMask`]: a gate of **content addresses**
//! (murmur3 seed-0 hashes), not token bytes. The LUT stays append-only;
//! departure is a lattice operation, never a LUT deletion.

use hllset_core::core::hashing::murmur3_hash;
use hllset_core::HLLSet;

use crate::kstorage::KStorage;
use crate::token_mask::TokenMask;

/// The incremental context vocabulary over a K-storage.
#[derive(Clone, Debug)]
pub struct ContextVocabulary<S: KStorage> {
    storage: S,
    mask: TokenMask,
}

/// One churn step: what was added to and removed from the vocabulary.
#[derive(Clone, Debug, Default)]
pub struct VocabDelta {
    /// Tokens whose content address entered the mask via `N`.
    pub added: Vec<Vec<u8>>,
    /// Content addresses removed via `D`.
    pub removed: Vec<u64>,
}

impl<S: KStorage> ContextVocabulary<S> {
    /// An empty vocabulary over `storage` (no tokens gated yet).
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            mask: TokenMask::new(),
        }
    }

    /// The underlying K-storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Number of gated tokens.
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Whether a token is currently in the vocabulary.
    pub fn contains(&self, token: &[u8]) -> bool {
        self.mask.contains_token(token)
    }

    /// Apply one `D/R/N` churn step: remove `M(departed)`, add `M(new)`.
    pub fn step(&mut self, departed: &HLLSet, new: &HLLSet) -> VocabDelta {
        let mut delta = VocabDelta::default();

        for token in self.storage.candidates(departed) {
            let hash = murmur3_hash(&token);
            if self.mask.remove(hash) {
                delta.removed.push(hash);
            }
        }

        for token in self.storage.candidates(new) {
            let hash = murmur3_hash(&token);
            if self.mask.insert(hash) {
                delta.added.push(token);
            }
        }

        delta
    }

    /// Register a token in both the storage and the mask (builder path).
    pub fn insert(&mut self, token: Vec<u8>) -> bool {
        self.storage.register(token.clone());
        self.mask.insert(murmur3_hash(&token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kstorage::TokenLutStorage;

    #[test]
    fn step_moves_vocabulary_with_drn() {
        let storage = TokenLutStorage::from_tokens(&["alpha", "beta", "gamma"]);
        let mut vocab = ContextVocabulary::new(storage);

        // V(0) = {alpha, beta}
        assert!(vocab.insert(b"alpha".to_vec()));
        assert!(vocab.insert(b"beta".to_vec()));
        assert_eq!(vocab.len(), 2);

        // N = {gamma}, D = {alpha}: alpha leaves, gamma enters.
        let new = HLLSet::from_tokens(&["gamma"]);
        let departed = HLLSet::from_tokens(&["alpha"]);
        let delta = vocab.step(&departed, &new);

        assert!(delta.removed.contains(&murmur3_hash(b"alpha")));
        assert!(delta.added.iter().any(|t| t == b"gamma"));
        assert!(!vocab.contains(b"alpha"));
        assert!(vocab.contains(b"beta"));
        assert!(vocab.contains(b"gamma"));
        assert_eq!(vocab.len(), 2);
    }
}
