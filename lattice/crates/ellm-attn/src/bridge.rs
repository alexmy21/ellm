//! `KBridge` — routes tokens through HLLSet K-storage into key vectors.
//!
//! The bridge is the single point of contact between the token realm and the
//! HLLSet realm. The transformer never hashes tokens inside the model graph;
//! it asks the bridge for a key, and the bridge consults the storage.

use crate::bit_table::BitKeyTable;
use crate::kstorage::{KStorage, KeyRef};

/// Routes tokens through a [`KStorage`] into key vectors.
#[derive(Clone, Debug)]
pub struct KBridge<S: KStorage> {
    storage: S,
}

impl<S: KStorage> KBridge<S> {
    /// Wrap a K-storage.
    pub fn new(storage: S) -> Self {
        Self { storage }
    }

    /// The underlying K-storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// The key reference for a token (K direction).
    pub fn key_ref(&self, token: &[u8]) -> KeyRef {
        self.storage.key_of(token)
    }

    /// Mode A address-key: the key vector of a token.
    ///
    /// Tier 1 keys read their single cell from the table; tier 2 keys return
    /// the centroid of their multi-seed cells.
    pub fn key_vector(&self, token: &[u8], table: &BitKeyTable) -> Vec<f32> {
        let d = table.dim();
        let mut out = vec![0.0f32; d];
        let cells = self.key_ref(token).cells();
        for cell in cells {
            let row = table.get(cell);
            for (out_i, &v) in row.iter().enumerate() {
                out[out_i] += v;
            }
        }
        let count = self.key_ref(token).cells().len().max(1) as f32;
        for v in &mut out {
            *v /= count;
        }
        out
    }

    /// Key vector plus a residual (positional/contextual), added in place.
    pub fn key_vector_with(
        &self,
        token: &[u8],
        table: &BitKeyTable,
        residual: &[f32],
    ) -> Vec<f32> {
        let mut v = self.key_vector(token, table);
        for (out_i, &r) in v.iter_mut().zip(residual) {
            *out_i += r;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_table::BitKeyTable;
    use crate::kstorage::TokenLutStorage;

    #[test]
    fn key_vector_is_cell_row() {
        let storage = TokenLutStorage::from_tokens(&["alpha"]);
        let bridge = KBridge::new(storage);
        let table = BitKeyTable::from_fn(4, |bit, dim| (bit * 10 + dim) as f32);
        let cell = bridge.key_ref(b"alpha").cells()[0];
        assert_eq!(bridge.key_vector(b"alpha", &table), table.get(cell));
    }

    #[test]
    fn key_vector_with_adds_residual() {
        let storage = TokenLutStorage::from_tokens(&["alpha"]);
        let bridge = KBridge::new(storage);
        let table = BitKeyTable::new(4);
        let out = bridge.key_vector_with(b"alpha", &table, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
