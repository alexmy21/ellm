//! MoE / Expert Think Tank — the expert ranking definitions.
//!
//! ```text
//! π    = sort_by_sha1( { Sᵢ : recent } )      // IICA shuffle, time-unbiased
//! F_k  = { S_π(i) : i ∈ [k·s, k·s + w) }      // convolution frame
//! E_k  = ∪ F_k                                // Expert
//! ρ(E) = BSSτ(E, R(t))                        // relevance to recent observation
//! ETT  = { (E_k, ρ_k) } sorted by ρ desc
//! EL   = argmax ρ
//! F(t) = EL ∪ { E_k : ρ_k ≥ τ_min }           // focused working set (bitmap)
//! ```
//!
//! Frames are formed over the original observations after a content-address
//! shuffle, so grouping is content-driven, not time-driven.

use hllset_core::core::hashing::sha1_hex;
use hllset_core::HLLSet;

use crate::setkey::bss_coverage;

/// One original observation with its content address.
#[derive(Clone, Debug)]
pub struct Observation {
    pub hllset: HLLSet,
    /// SHA1 of the serialized HLLSet (the `h:` content key material).
    pub cid: String,
}

impl Observation {
    pub fn new(hllset: HLLSet) -> Self {
        let cid = sha1_hex(&hllset.to_bytes());
        Self { hllset, cid }
    }
}

/// One expert: the union of a convolution frame.
#[derive(Clone, Debug)]
pub struct Expert {
    pub hllset: HLLSet,
    /// Content address of the expert union.
    pub cid: String,
    /// Original observation indices that fell into this frame.
    pub frame: Vec<usize>,
}

/// An expert with its relevance to the recent observation.
#[derive(Clone, Debug)]
pub struct RankedExpert {
    pub expert: Expert,
    pub relevance: f64,
}

/// Content-address shuffle: observation indices sorted by SHA1 cid.
pub fn shuffle_by_sha1(observations: &[Observation]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..observations.len()).collect();
    indices.sort_by(|&a, &b| observations[a].cid.cmp(&observations[b].cid));
    indices
}

/// Convolution frames over shuffled indices: frame `k = [k·s, k·s + w)`.
pub fn frames(indices: &[usize], stride: usize, window: usize) -> Vec<Vec<usize>> {
    if window == 0 || indices.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut k = 0;
    while k * stride < indices.len() {
        let start = k * stride;
        let end = (start + window).min(indices.len());
        out.push(indices[start..end].to_vec());
        if end == indices.len() {
            break;
        }
        k += 1;
    }
    out
}

/// Build experts as unions of convolution frames.
pub fn build_experts(
    observations: &[Observation],
    indices: &[usize],
    stride: usize,
    window: usize,
) -> Vec<Expert> {
    frames(indices, stride, window)
        .into_iter()
        .map(|frame| {
            let mut hllset = HLLSet::new();
            for &i in &frame {
                hllset = hllset.union(&observations[i].hllset);
            }
            let cid = sha1_hex(&hllset.to_bytes());
            Expert { hllset, cid, frame }
        })
        .collect()
}

/// Rank experts by BSS relevance to the recent observation (descending).
pub fn rank(recent: &HLLSet, experts: &[Expert]) -> Vec<RankedExpert> {
    let mut ranked: Vec<RankedExpert> = experts
        .iter()
        .map(|expert| RankedExpert {
            relevance: bss_coverage(&expert.hllset, recent),
            expert: expert.clone(),
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

/// The focused working set: the top expert plus every expert at or above
/// `tau_min` relevance.
pub fn focus(ranked: &[RankedExpert], tau_min: f64) -> HLLSet {
    let mut out = HLLSet::new();
    for (i, ranked_expert) in ranked.iter().enumerate() {
        if i == 0 || ranked_expert.relevance >= tau_min {
            out = out.union(&ranked_expert.expert.hllset);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(tokens: &[&str]) -> Observation {
        Observation::new(HLLSet::from_tokens(tokens))
    }

    #[test]
    fn shuffle_is_deterministic_and_content_addressed() {
        let observations = vec![
            obs(&["alpha"]),
            obs(&["beta"]),
            obs(&["gamma"]),
        ];
        let a = shuffle_by_sha1(&observations);
        let b = shuffle_by_sha1(&observations);
        assert_eq!(a, b);
    }

    #[test]
    fn frames_partition_indices() {
        let indices = shuffle_by_sha1(&[obs(&["a"]), obs(&["b"]), obs(&["c"])]);
        let fs = frames(&indices, 2, 2);
        assert!(!fs.is_empty());
        for frame in &fs {
            assert!(!frame.is_empty());
        }
        let all: Vec<usize> = fs.iter().flatten().copied().collect();
        let mut sorted = all.clone();
        sorted.sort_unstable();
        let mut expected: Vec<usize> = (0..3).collect();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn rank_puts_best_expert_first_and_focus_unions() {
        let observations = vec![obs(&["alpha", "beta"]), obs(&["gamma"])];
        let indices = shuffle_by_sha1(&observations);
        let experts = build_experts(&observations, &indices, 1, 1);
        let recent = HLLSet::from_tokens(&["alpha"]);
        let ranked = rank(&recent, &experts);
        assert!(!ranked.is_empty());
        assert!(ranked[0].relevance >= ranked[ranked.len() - 1].relevance);

        let focused = focus(&ranked, 0.5);
        assert!(focused.popcount() > 0);
    }
}
