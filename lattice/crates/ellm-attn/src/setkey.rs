//! HLLSet-native attention scores (Mode B).
//!
//! These are the lattice morphisms that replace the learned dot product when
//! attention runs entirely in the HLLSet realm: `Q = I(query)`,
//! `K = context sub-lattice`, score = a set morphism, `V = M(K ∩ Q)`.

use hllset_core::HLLSet;

/// Overlap popcount: `|A ∩ B|` — exact bit count, no cardinality estimate.
pub fn overlap(a: &HLLSet, b: &HLLSet) -> f64 {
    a.intersection(b).popcount() as f64
}

/// Jaccard similarity: `|A ∩ B| / |A ∪ B|`.
pub fn jaccard(a: &HLLSet, b: &HLLSet) -> f64 {
    a.jaccard_similarity(b)
}

/// BSS coverage: how much of `query` is covered by `context`.
///
/// This is the directed score natural for attention: a context key covers a
/// query when most of the query's bits are already present in the context.
/// Computed as `context.bss_inclusion(query) = |context ∩ query| / |query|`.
pub fn bss_coverage(context: &HLLSet, query: &HLLSet) -> f64 {
    context.bss_inclusion(query)
}

/// Symmetric BSS: `min(τ(A,B), τ(B,A))` — both directions must hold.
pub fn bss_symmetric(a: &HLLSet, b: &HLLSet) -> f64 {
    a.bss_inclusion(b).min(b.bss_inclusion(a))
}

/// One retrieved candidate set: the lattice score and the materialized
/// tokens behind the `K ∩ Q` intersection.
#[derive(Clone, Debug)]
pub struct Retrieval {
    /// Lattice score between the query and the context sub-lattice.
    pub score: f64,
    /// Materialized candidate tokens (the V side).
    pub candidates: Vec<Vec<u8>>,
}

/// Retrieve candidates from a context sub-lattice for a query HLLSet.
///
/// The score is BSS coverage of the query by the context; the candidates are
/// the tokens materialized from `context ∩ query`.
pub fn retrieve(context: &HLLSet, query: &HLLSet, candidates: Vec<Vec<u8>>) -> Retrieval {
    Retrieval {
        score: bss_coverage(context, query),
        candidates,
    }
}

/// Batch BSS-coverage scoring of one query against many context nodes.
///
/// This is the hot path for top-HLLSet ranking (waypoint selection and the
/// dispatcher): one query vs the candidate `H(t)` nodes.
pub fn score_many(query: &HLLSet, contexts: &[HLLSet]) -> Vec<f64> {
    contexts.iter().map(|c| bss_coverage(c, query)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_is_exact_popcount() {
        let a = HLLSet::from_tokens(&["alpha"]);
        let b = HLLSet::from_tokens(&["alpha", "beta"]);
        assert_eq!(overlap(&a, &b), a.popcount() as f64);
    }

    #[test]
    fn jaccard_is_one_for_identical() {
        let a = HLLSet::from_tokens(&["alpha", "beta"]);
        assert!((jaccard(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bss_coverage_is_one_when_context_covers_query() {
        let query = HLLSet::from_tokens(&["alpha"]);
        let context = HLLSet::from_tokens(&["alpha", "beta", "gamma"]);
        assert!((bss_coverage(&context, &query) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bss_symmetric_is_bounded() {
        let a = HLLSet::from_tokens(&["alpha", "beta"]);
        let b = HLLSet::from_tokens(&["beta", "gamma"]);
        let s = bss_symmetric(&a, &b);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn score_many_matches_scalar() {
        let q = HLLSet::from_tokens(&["alpha"]);
        let contexts = vec![
            HLLSet::from_tokens(&["alpha", "beta"]),
            HLLSet::from_tokens(&["gamma"]),
        ];
        let scores = score_many(&q, &contexts);
        assert_eq!(scores[0], bss_coverage(&contexts[0], &q));
        assert_eq!(scores[1], bss_coverage(&contexts[1], &q));
    }
}
