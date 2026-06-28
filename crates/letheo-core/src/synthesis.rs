//! Synthesis layer — the "dream" / semantic compression (`DISTILL`).
//!
//! Takes N perceptions, computes their **centroid** and measures the **semantic variance** as the
//! cosine similarity of each event to the centroid:
//!
//! - close to the centroid (`sim ≥ θ_redundancy`) → redundant noise → FADE.
//! - the centroid itself                           → the user's direction → retain.
//! - outlier (`sim ≤ θ_anomaly`)                   → novelty / abrupt change → retain.

use crate::modes::{cluster_modes, ModeConfig, ModeSeed};
use crate::perception::Perception;
use crate::vector::{centroid_refs, cosine, Vector};

/// Threshold above which an event is considered redundant (predictable) → FADE.
pub const DEFAULT_THETA_REDUNDANCY: f32 = 0.92;
/// Threshold below which an event is an anomalous outlier (novelty) → retain.
pub const DEFAULT_THETA_ANOMALY: f32 = 0.30;

/// The product of the dream: an Intention Vector that compresses many perceptions.
#[derive(Debug, Clone)]
pub struct IntentionVector {
    pub subject: String,
    /// The central direction of the user's behaviour (the centroid).
    pub centroid: Vector,
    /// Retained anomalous embeddings (novelty / pattern breaks).
    pub anomalies: Vec<Vector>,
    /// Representative text of the **most central** perception (the most typical of the cluster). It is
    /// the lexical label of the core, so the prose names the dominant content.
    pub core_label: String,
    /// Lexical labels of each anomaly, aligned with `anomalies`.
    pub anomaly_labels: Vec<String>,
    /// How many perceptions collapsed into this vector (for the compression ratio).
    pub absorbed: usize,
    /// How many were marked as redundant noise (FADE candidates).
    pub redundant: usize,
    /// Lexical-label histogram of the cycle: `(text, count)` sorted by frequency desc. Lets us
    /// reconstruct **per-behaviour** trajectories along the arc (not just the global centroid) →
    /// answer "did X come back?" for a concrete behaviour.
    pub label_histogram: Vec<(String, usize)>,
    /// **Modes** of the cycle: coherent behaviour subgroups (deterministic clustering). The
    /// `centroid` above is the GLOBAL mean (arc origin, backwards-compatible); the modes are the
    /// multi-modal decomposition that keeps the mean from collapsing distinct behaviours into noise.
    /// Unimodal ⇒ a single mode ≈ the global centroid. See [`crate::modes`].
    pub modes: Vec<ModeSeed>,
}

/// `DISTILL` parameters.
#[derive(Debug, Clone, Copy)]
pub struct DistillConfig {
    pub theta_redundancy: f32,
    pub theta_anomaly: f32,
    /// Multi-modal clustering parameters (see [`crate::modes`]).
    pub modes: ModeConfig,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            theta_redundancy: DEFAULT_THETA_REDUNDANCY,
            theta_anomaly: DEFAULT_THETA_ANOMALY,
            modes: ModeConfig::default(),
        }
    }
}

/// `DISTILL`: collapses perceptions into an Intention Vector.
///
/// Returns `None` if there are no perceptions or their dimensions do not match.
pub fn distill(
    subject: &str,
    perceptions: &[&Perception],
    cfg: DistillConfig,
) -> Option<IntentionVector> {
    if perceptions.is_empty() {
        return None;
    }
    // Centroid **without cloning** the embeddings: they are referenced (it used to clone a
    // `Vec<Vector>` of N×dim floats per dream, just to average).
    let refs: Vec<&[f32]> = perceptions.iter().map(|p| p.embedding.as_slice()).collect();
    let c = centroid_refs(&refs)?;

    let mut anomalies = Vec::new();
    let mut anomaly_labels = Vec::new();
    let mut redundant = 0usize;
    // The core is labelled by the **mode** (the most frequent behaviour), not by the perception
    // closest to the centroid: with a mixed centroid, the "closest" can be an unrepresentative event.
    // The mode reflects what *dominates*.
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in perceptions.iter() {
        *freq.entry(p.representative_text()).or_insert(0) += 1;
        let sim = cosine(&p.embedding, &c);
        if sim >= cfg.theta_redundancy {
            redundant += 1; // predictable noise → its vote already lives in the centroid → FADE
        } else if sim <= cfg.theta_anomaly {
            anomalies.push(p.embedding.clone()); // novelty → only what is RETAINED is cloned
            anomaly_labels.push(p.representative_text());
        }
    }
    // Histogram sorted by frequency desc (deterministic alphabetical tiebreak).
    let mut label_histogram: Vec<(String, usize)> = freq.into_iter().collect();
    label_histogram.sort_by(|(ta, ca), (tb, cb)| cb.cmp(ca).then_with(|| ta.cmp(tb)));
    // Mode = most frequent label (head of the histogram).
    let core_label = label_histogram
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();

    // Multi-modal decomposition: coherent subgroups instead of a single mean. If the set is
    // unimodal, this returns a single mode ≈ the global centroid (backwards-compatible).
    let modes = cluster_modes(perceptions, cfg.modes);

    Some(IntentionVector {
        subject: subject.to_string(),
        centroid: c,
        anomalies,
        core_label,
        anomaly_labels,
        absorbed: perceptions.len(),
        redundant,
        label_histogram,
        modes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::Perception;

    const HF: f64 = 3600.0;

    fn p(subject: &str, e: Vec<f32>) -> Perception {
        Perception::new(subject, e, 1.0, HF, 0.0)
    }

    #[test]
    fn distill_empty_is_none() {
        assert!(distill("user:X", &[], DistillConfig::default()).is_none());
    }

    #[test]
    fn distill_captures_core_and_anomaly_labels() {
        // Dense cluster labelled "trail" + a "crypto" outlier. The core must be labelled with the
        // central content and the anomaly keep its text.
        let trail: Vec<Perception> = (0..8)
            .map(|i| p("u", vec![1.0, 0.0 + i as f32 * 0.001]).with_trait("act", "trail"))
            .collect();
        let outlier = p("u", vec![0.0, 1.0]).with_trait("act", "crypto");
        let mut refs: Vec<&Perception> = trail.iter().collect();
        refs.push(&outlier);
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        assert_eq!(
            iv.core_label, "trail",
            "the core is labelled with the central content"
        );
        assert!(
            iv.anomaly_labels.contains(&"crypto".to_string()),
            "the anomaly keeps its text"
        );
        assert_eq!(
            iv.anomaly_labels.len(),
            iv.anomalies.len(),
            "labels aligned with vectors"
        );
    }

    #[test]
    fn redundant_cluster_collapses_to_centroid() {
        // Five nearly identical events: high redundancy, centroid in their direction.
        let ps = [
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![0.99, 0.01]),
            p("user:X", vec![1.0, 0.02]),
            p("user:X", vec![0.98, 0.0]),
            p("user:X", vec![1.0, 0.0]),
        ];
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("user:X", &refs, DistillConfig::default()).unwrap();
        assert_eq!(iv.absorbed, 5);
        assert!(
            iv.redundant >= 4,
            "the dense cluster is mostly redundant"
        );
        assert!(
            iv.anomalies.is_empty(),
            "no novelty in a homogeneous cluster"
        );
    }

    #[test]
    fn outlier_is_retained_as_anomaly() {
        // Four events in one direction + an orthogonal outlier (abrupt behaviour change).
        let ps = [
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![0.0, 1.0]), // outlier
        ];
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("user:X", &refs, DistillConfig::default()).unwrap();
        assert_eq!(iv.anomalies.len(), 1, "the outlier is retained as novelty");
    }

    #[test]
    fn compression_ratio_is_meaningful() {
        let ps: Vec<Perception> = (0..1000).map(|_| p("user:X", vec![1.0, 0.0])).collect();
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("user:X", &refs, DistillConfig::default()).unwrap();
        // 1000 perceptions → 1 centroid (+ 0 anomalies): massive compression.
        assert_eq!(iv.absorbed, 1000);
        assert!(iv.anomalies.is_empty());
    }
}
