//! `MockProvider` — deterministic embeddings without a model.
//!
//! Lets us validate the physics and the parser in offline CI. The same text always produces the same
//! vector; similar texts produce similar vectors (hashing tokens into buckets).

use crate::provider::{Provider, EMBED_DIM};

/// Deterministic provider based on token hashing. No dependencies, no network.
#[derive(Debug, Clone)]
pub struct MockProvider {
    dim: usize,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self { dim: EMBED_DIM }
    }
}

impl MockProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allows a small dimension in tests.
    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }
}

/// 64-bit FNV-1a hash — deterministic and dependency-free.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Provider for MockProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        // Bag-of-tokens: each token increments a deterministic bucket. Similar texts → similar vectors
        // (they share tokens), enough to test resonance and centroids.
        for token in text.split_whitespace() {
            let h = fnv1a(&token.to_lowercase());
            let bucket = (h % self.dim as u64) as usize;
            let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[bucket] += sign;
        }
        // Normalize to a unit vector (stabilizes the cosine).
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let p = MockProvider::new();
        assert_eq!(p.embed("running shoes"), p.embed("running shoes"));
    }

    #[test]
    fn similar_texts_more_similar_than_different() {
        let p = MockProvider::with_dim(64);
        let a = p.embed("nighttime running shoes");
        let b = p.embed("morning running shoes");
        let c = p.embed("life insurance mortgage bank");
        let cos = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(i, j)| i * j).sum::<f32>();
        assert!(cos(&a, &b) > cos(&a, &c), "related texts resonate more");
    }

    #[test]
    fn correct_dim() {
        assert_eq!(MockProvider::new().embed("x").len(), EMBED_DIM);
    }
}
