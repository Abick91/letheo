//! # letheo-index · ANN index (HNSW) over the engine
//!
//! The core does linear Flat search (exact, O(n)) — perfect at tens of modes/facts, impractical at
//! millions. This crate provides the acceleration: an **HNSW** index over both layers —
//! `(subject × mode)` (layer-2) and facts (layer-1) — queried in O(log n). A **separate crate** so
//! `letheo-core` stays hermetic/offline (just as `letheo-async` isolates Tokio): the external ANN
//! dependency does not enter the core.
//!
//! **Metric trick**: HNSW assumes a metric distance; the engine ranks by **cosine**. On unit-norm
//! vectors, Euclidean distance is monotone with `(1 − cosine)` (`‖a−b‖² = 2 − 2·cos`), so the
//! nearest Euclidean neighbour *is* the one with highest cosine. Hence the index normalises
//! centroids and queries. The core's Flat is kept as the **exact oracle**.
//!
//! The [`Retriever`] wires both together: exact Flat below a size threshold, HNSW above, with
//! integrated **life-filtering** (filtered-ANN).

use instant_distance::{Builder, HnswMap, Point, Search};
use letheo_core::{ArchetypeStore, FactStore, Tick};
use std::collections::HashSet;

/// Fixed HNSW builder seed → **bit-for-bit reproducible** index (no system randomness), consistent
/// with the engine's determinism discipline.
const INDEX_SEED: u64 = 0x1E7E0;

/// An index point: an embedding **normalised to unit norm**.
#[derive(Clone, Debug)]
struct UnitPoint(Vec<f32>);

impl Point for UnitPoint {
    /// Euclidean distance. On unit vectors it is monotone with `(1 − cosine)`, so the ANN ranking
    /// matches the engine's cosine ranking.
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(&other.0)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }
}

/// Which `(subject, mode)` a layer-2 point corresponds to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModeRef {
    pub subject: String,
    /// Index of the mode within the archetype (`0` if the archetype has no modes: its global core).
    pub mode: usize,
}

/// Which fact a layer-1 point corresponds to: the subject and the fact's position at index build time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FactRef {
    pub subject: String,
    pub position: usize,
}

/// Generic ANN index over `(embedding, value)` points. `ModeIndex`/`FactIndex` specialise it.
pub struct AnnIndex<V> {
    map: Option<HnswMap<UnitPoint, V>>,
    len: usize,
}

impl<V: Clone> AnnIndex<V> {
    /// Builds the index from raw `(embedding, value)` points.
    pub fn from_points(points: Vec<(Vec<f32>, V)>) -> Self {
        let len = points.len();
        if len == 0 {
            return Self { map: None, len: 0 };
        }
        let (pts, vals): (Vec<UnitPoint>, Vec<V>) = points
            .into_iter()
            .map(|(c, v)| (UnitPoint(normalize(&c)), v))
            .unzip();
        let map = Builder::default().seed(INDEX_SEED).build(pts, vals);
        Self {
            map: Some(map),
            len,
        }
    }

    /// The `k` values whose embedding resonates most with the query, via HNSW (≈ O(log n)).
    /// Equivalent to Flat top-k by cosine, without scanning all points.
    pub fn resonate(&self, query: &[f32], k: usize) -> Vec<V> {
        let map = match &self.map {
            Some(m) => m,
            None => return Vec::new(),
        };
        let q = UnitPoint(normalize(query));
        let mut search = Search::default();
        map.search(&q, &mut search)
            .take(k)
            .map(|item| item.value.clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Index of **layer-2**: one point per `(subject × mode)` in an `ArchetypeStore`.
pub type ModeIndex = AnnIndex<ModeRef>;

impl ModeIndex {
    /// Builds the index from a store: one point per mode of each archetype (centroid). A legacy
    /// archetype with no modes contributes its global core as a single point (`mode = 0`).
    pub fn build(store: &ArchetypeStore) -> Self {
        let mut points = Vec::new();
        for a in store.iter() {
            if a.modes.is_empty() {
                points.push((
                    a.core.clone(),
                    ModeRef {
                        subject: a.subject.clone(),
                        mode: 0,
                    },
                ));
            } else {
                for (i, m) in a.modes.iter().enumerate() {
                    points.push((
                        m.centroid.clone(),
                        ModeRef {
                            subject: a.subject.clone(),
                            mode: i,
                        },
                    ));
                }
            }
        }
        Self::from_points(points)
    }

    /// The `k` most resonant **subjects** (deduplicating modes from the same subject, preserving order).
    pub fn resonate_subjects(&self, query: &[f32], k: usize) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for r in self.resonate(query, k.saturating_mul(4).max(k)) {
            if seen.insert(r.subject.clone()) {
                out.push(r.subject);
                if out.len() == k {
                    break;
                }
            }
        }
        out
    }
}

/// Index of **layer-1**: one point per episodic fact in a `FactStore`.
pub type FactIndex = AnnIndex<FactRef>;

impl FactIndex {
    /// Builds the index from episodic memory: one point per fact (its embedding).
    pub fn build_facts(store: &FactStore) -> Self {
        let points = store
            .iter()
            .enumerate()
            .map(|(i, f)| {
                (
                    f.embedding.clone(),
                    FactRef {
                        subject: f.subject.clone(),
                        position: i,
                    },
                )
            })
            .collect();
        Self::from_points(points)
    }
}

/// Retriever combining the core's **exact Flat** (fast at small scale, ranks by relevance·life)
/// with **HNSW** (at large scale), switching on a size threshold. Caches the index and rebuilds
/// it when the archetype count changes. Filters HNSW results by **life** (filtered-ANN) so a
/// faded archetype is not returned even if it resonates.
pub struct Retriever {
    threshold: usize,
    cache: Option<(usize, ModeIndex)>,
}

impl Retriever {
    /// Creates a retriever that uses Flat up to `threshold` archetypes and HNSW above.
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            cache: None,
        }
    }

    /// Top-`k` resonant **live** subjects. Below the threshold uses the core's exact Flat (which
    /// already ranks by relevance·life); above it uses HNSW (rebuilt if the store size changed) and
    /// filters by life. `&mut self` because it caches the index.
    pub fn resonate_subjects(
        &mut self,
        store: &ArchetypeStore,
        query: &[f32],
        k: usize,
        now: Tick,
        theta_fade: f64,
    ) -> Vec<String> {
        if store.len() <= self.threshold {
            return store
                .resonate(query, k, now, theta_fade)
                .into_iter()
                .map(|a| a.subject.clone())
                .collect();
        }
        // HNSW: (re)build if the archetype count changed.
        if !matches!(&self.cache, Some((n, _)) if *n == store.len()) {
            self.cache = Some((store.len(), ModeIndex::build(store)));
        }
        let index = &self.cache.as_ref().unwrap().1;

        // Wide candidate margin → filter by life → deduplicate to k subjects.
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for r in index.resonate(query, k.saturating_mul(8).max(k)) {
            if let Some(a) = store.get(&r.subject) {
                if a.trace.weight(now) >= theta_fade && seen.insert(r.subject.clone()) {
                    out.push(r.subject);
                    if out.len() == k {
                        break;
                    }
                }
            }
        }
        out
    }

    /// Invalidates the cached index (use after mutating modes without changing the archetype count).
    pub fn invalidate(&mut self) {
        self.cache = None;
    }
}

/// Normalises to unit norm. Zero vector → returned as-is (distance remains well-defined).
fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_core::synthesis::IntentionVector;
    use letheo_core::{ArchetypeStore, FactStore, Resilience};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Deterministic pseudo-random vector (LCG) of dimension `dim`.
    fn pseudo(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37);
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// `recall@k` of the index against the exact Flat oracle, averaged over `queries` queries.
    fn measure_recall(points: &[(Vec<f32>, ModeRef)], dim: usize, k: usize, queries: u64) -> f64 {
        let index = AnnIndex::from_points(points.to_vec());
        let (mut hits, mut total) = (0usize, 0usize);
        for q in 0..queries {
            let query = pseudo(dim, 1_000_000 + q);
            let mut exact: Vec<(f32, &ModeRef)> =
                points.iter().map(|(c, r)| (cosine(c, &query), r)).collect();
            exact.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let exact_set: HashSet<&ModeRef> = exact.iter().take(k).map(|(_, r)| *r).collect();
            hits += index
                .resonate(&query, k)
                .iter()
                .filter(|r| exact_set.contains(r))
                .count();
            total += k;
        }
        hits as f64 / total as f64
    }

    #[test]
    fn ann_recall_at_k_matches_flat_oracle() {
        let dim = 48;
        let points: Vec<(Vec<f32>, ModeRef)> = (0..800)
            .map(|i| {
                (
                    pseudo(dim, i as u64),
                    ModeRef {
                        subject: format!("s{i}"),
                        mode: 0,
                    },
                )
            })
            .collect();
        let recall = measure_recall(&points, dim, 10, 60);
        assert!(
            recall >= 0.99,
            "recall@10 = {recall:.4} (target ≥ 0.99 vs exact Flat)"
        );
    }

    #[test]
    #[ignore = "scale (20k modes): HNSW build is slow in debug; run with --ignored or --release"]
    fn ann_holds_recall_at_scale() {
        // At 20,000 modes the index builds and maintains high recall (engine at scale).
        let dim = 48;
        let points: Vec<(Vec<f32>, ModeRef)> = (0..20_000)
            .map(|i| {
                (
                    pseudo(dim, i as u64),
                    ModeRef {
                        subject: format!("s{i}"),
                        mode: 0,
                    },
                )
            })
            .collect();
        let recall = measure_recall(&points, dim, 10, 30);
        assert!(recall >= 0.95, "recall@10 at scale = {recall:.4} (≥ 0.95)");
    }

    #[test]
    fn fact_index_recovers_the_relevant_fact() {
        // Layer-1: facts with distinct directions → index retrieves the correct one per query.
        let mut fs = FactStore::new();
        fs.remember(
            "u0",
            "peanuts allergy",
            vec![1.0, 0.0, 0.0],
            "t",
            1.0,
            86_400.0,
            0.0,
        );
        fs.remember(
            "u1",
            "red car",
            vec![0.0, 1.0, 0.0],
            "t",
            1.0,
            86_400.0,
            0.0,
        );
        fs.remember(
            "u2",
            "window seat",
            vec![0.0, 0.0, 1.0],
            "t",
            1.0,
            86_400.0,
            0.0,
        );
        let index = FactIndex::build_facts(&fs);
        assert_eq!(index.len(), 3);
        let top = index.resonate(&[0.1, 0.9, 0.0], 1);
        assert_eq!(
            top[0],
            FactRef {
                subject: "u1".into(),
                position: 1
            }
        );
    }

    fn store_of(n: usize, dim: usize) -> ArchetypeStore {
        let mut store = ArchetypeStore::new();
        for i in 0..n {
            store.imprint(
                &IntentionVector {
                    subject: format!("user:{i}"),
                    centroid: pseudo(dim, i as u64),
                    anomalies: vec![],
                    core_label: "x".into(),
                    anomaly_labels: vec![],
                    absorbed: 10,
                    redundant: 0,
                    label_histogram: vec![("x".into(), 10)],
                    modes: vec![],
                },
                Resilience::High,
                0.0,
            );
        }
        store
    }

    #[test]
    fn retriever_flat_and_hnsw_paths_agree_on_top_subject() {
        let dim = 48;
        let store = store_of(120, dim);
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        let mut flat = Retriever::new(10_000); // 120 ≤ 10000 → Flat
        let mut hnsw = Retriever::new(0); //               120 > 0     → HNSW
        for q in 0..20 {
            let query = pseudo(dim, 7_000 + q);
            let f = flat.resonate_subjects(&store, &query, 1, 0.0, theta);
            let h = hnsw.resonate_subjects(&store, &query, 1, 0.0, theta);
            assert_eq!(f, h, "Flat and HNSW agree on top subject (q={q})");
        }
    }

    #[test]
    fn retriever_hnsw_filters_out_faded_subjects() {
        // A highly relevant but faded subject is NOT returned via the HNSW path (filtered-ANN).
        let dim = 8;
        let mut store = store_of(60, dim);
        // Insert a subject aligned with the query but with a short half-life → faded by T.
        let target = pseudo(dim, 999);
        store.imprint(
            &IntentionVector {
                subject: "user:faded".into(),
                centroid: target.clone(),
                anomalies: vec![],
                core_label: "x".into(),
                anomaly_labels: vec![],
                absorbed: 10,
                redundant: 0,
                label_histogram: vec![("x".into(), 10)],
                modes: vec![],
            },
            Resilience::Low,
            0.0,
        );
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        let mut hnsw = Retriever::new(0);
        // At 200 days the Low subject (half-life 30d) has faded, while High (720d) subjects are
        // still alive. The faded one does not appear despite perfect resonance (filtered-ANN).
        let later = 200.0 * 86_400.0;
        let hits = hnsw.resonate_subjects(&store, &target, 5, later, theta);
        assert!(!hits.is_empty(), "there are live subjects to return");
        assert!(
            !hits.contains(&"user:faded".to_string()),
            "faded subject was filtered: {hits:?}"
        );
    }

    #[test]
    fn empty_index_is_well_defined() {
        let index: AnnIndex<ModeRef> = AnnIndex::from_points(Vec::new());
        assert!(index.is_empty());
        assert!(index.resonate(&[1.0, 0.0], 5).is_empty());
    }
}
