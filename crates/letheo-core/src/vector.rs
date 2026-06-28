//! Minimal vector operations for the Cognitive Runtime.
//!
//! Linear Flat search (cosine). An archetype holds dozens of Intention Vectors, not millions;
//! linear is exact and predictable at that scale. The ANN index (HNSW) arrives to scale to millions.

/// A dense vector (embedding / Intention Vector). We use 384 dims (all-MiniLM-L6-v2),
/// but the type is dimension-agnostic.
pub type Vector = Vec<f32>;

/// Vectorization width. 8 `f32` = 256 bits (AVX). We use independent accumulators to break the
/// reduction dependency chain and let LLVM emit packed SIMD instructions — no external deps and no
/// `unsafe`, portable to any stable target.
const LANES: usize = 8;

/// Sum of squares (‖v‖²), vectorized. Basis of the norm; avoids the root when only comparing.
pub fn sq_norm(v: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut chunks = v.chunks_exact(LANES);
    for c in chunks.by_ref() {
        for l in 0..LANES {
            acc[l] += c[l] * c[l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for &x in chunks.remainder() {
        s += x * x;
    }
    s
}

/// Euclidean (L2) norm.
pub fn norm(v: &[f32]) -> f32 {
    sq_norm(v).sqrt()
}

/// Dot product, vectorized (per-lane accumulators + scalar tail).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "incompatible dimensions");
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..LANES {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        s += x * y;
    }
    s
}

/// Dot product and both sums of squares in **a single pass** — the hot operation of `cosine`, which
/// would otherwise traverse the vectors three times.
fn dot_and_sq_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot_acc = [0.0f32; LANES];
    let mut sa_acc = [0.0f32; LANES];
    let mut sb_acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..LANES {
            dot_acc[l] += x[l] * y[l];
            sa_acc[l] += x[l] * x[l];
            sb_acc[l] += y[l] * y[l];
        }
    }
    let mut d: f32 = dot_acc.iter().sum();
    let mut sa: f32 = sa_acc.iter().sum();
    let mut sb: f32 = sb_acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        d += x * y;
        sa += x * x;
        sb += y * y;
    }
    (d, sa, sb)
}

/// Cosine similarity in `[-1, 1]`. Returns 0.0 if either vector is null.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (d, sa, sb) = dot_and_sq_norms(a, b);
    if sa == 0.0 || sb == 0.0 {
        return 0.0;
    }
    d / (sa.sqrt() * sb.sqrt())
}

/// Centroid (mean vector) of a set. Basis of `semantic_variance` (see `synthesis.rs`).
/// Returns `None` if the input is empty or the dimensions do not match.
pub fn centroid(vectors: &[Vector]) -> Option<Vector> {
    let first = vectors.first()?;
    let dim = first.len();
    if vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v) {
            *a += *x;
        }
    }
    let n = vectors.len() as f32;
    for a in acc.iter_mut() {
        *a /= n;
    }
    Some(acc)
}

/// Centroid of a set of **referenced** vectors — avoids cloning the embeddings (the hot path of
/// `DISTILL`: it used to clone a `Vec<Vector>` of N×dim floats just to average).
/// Returns `None` if empty or the dimensions do not match.
pub fn centroid_refs(vectors: &[&[f32]]) -> Option<Vector> {
    let dim = vectors.first()?.len();
    if vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(*v) {
            *a += *x;
        }
    }
    let n = vectors.len() as f32;
    for a in acc.iter_mut() {
        *a /= n;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_null_vector_is_zero() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn centroid_is_mean() {
        let c = centroid(&[vec![0.0, 0.0], vec![2.0, 4.0]]).unwrap();
        assert_eq!(c, vec![1.0, 2.0]);
    }

    #[test]
    fn centroid_rejects_mismatched_dims() {
        assert!(centroid(&[vec![1.0], vec![1.0, 2.0]]).is_none());
    }

    // ── Equivalence of the vectorized versions with scalar references ───────────

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn naive_sq_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum()
    }

    /// Deterministic pseudo-random vector (LCG) of dimension `dim`.
    fn pseudo(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(1);
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn vectorized_matches_scalar_across_dims() {
        // Includes dimensions not multiples of LANES (8) to exercise the scalar tail.
        for &dim in &[0usize, 1, 7, 8, 9, 13, 16, 33, 384] {
            let a = pseudo(dim, 11);
            let b = pseudo(dim, 99);

            let d_ref = naive_dot(&a, &b);
            assert!(
                (dot(&a, &b) - d_ref).abs() <= 1e-3 + d_ref.abs() * 1e-5,
                "dot dim={dim}"
            );

            let sa_ref = naive_sq_norm(&a);
            assert!(
                (sq_norm(&a) - sa_ref).abs() <= 1e-3 + sa_ref * 1e-5,
                "sq_norm dim={dim}"
            );

            // cosine must match the direct reference formula.
            if dim > 0 {
                let nb = naive_sq_norm(&b);
                let cos_ref = if sa_ref == 0.0 || nb == 0.0 {
                    0.0
                } else {
                    d_ref / (sa_ref.sqrt() * nb.sqrt())
                };
                assert!((cosine(&a, &b) - cos_ref).abs() <= 1e-5, "cosine dim={dim}");
            }
        }
    }

    #[test]
    fn empty_vectors_are_well_defined() {
        assert_eq!(dot(&[], &[]), 0.0);
        assert_eq!(sq_norm(&[]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
    }
}
