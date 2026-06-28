//! Modes layer — the **multi-modal** archetype.
//!
//! A subject is rarely a single thing: they watch film noir *and* documentaries, code in Rust *and*
//! write prose. A single centroid (the mean of everything) collapses those distinct behaviours into
//! an intermediate point that **represents none of them** — the mean of "thriller" and "romantic
//! comedy" is not a genre, it's noise. That was the engine's #1 bottleneck: on multi-modal data the
//! signal was destroyed by averaging.
//!
//! Here behaviour is decomposed into **modes**: coherent subgroups of perceptions, each with its own
//! centroid, label and **independent forgetting physics** (a mode that is not revisited decays and
//! fades; one that recurs is reinforced). The clustering is **deterministic** (no RNG): a
//! *leader / DP-means* style assignment in arrival order — bit-for-bit reproducible, with no hidden
//! seeds (consistent with the TRUTH 100% discipline: no undeclared randomness).

use crate::entropy::{EntropyTrace, Tick};
use crate::perception::Perception;
use crate::vector::{cosine, Vector};
use std::collections::HashMap;

/// Cosine boundary between "the same behaviour mode" and "a different mode". Above the threshold, two
/// directions are considered the same mode (they merge); below it, a new mode is born. Calibratable,
/// declared physics (not a magic constant): with normalized embeddings (all-MiniLM), the same topic
/// sits around 0.6–0.9 and distinct topics 0.1–0.4, so 0.5 separates them comfortably.
pub const DEFAULT_MODE_THETA: f32 = 0.5;

/// Maximum number of modes per archetype in a distillation cycle. Bounds the cost and keeps noise
/// from fragmenting the identity into a thousand pieces. Declared, adjustable via [`ModeConfig`].
pub const DEFAULT_MAX_MODES: usize = 8;

/// Mode clustering parameters (part of `DistillConfig`).
#[derive(Debug, Clone, Copy)]
pub struct ModeConfig {
    /// Cosine threshold to assign to an existing mode vs create a new one (see [`DEFAULT_MODE_THETA`]).
    pub theta: f32,
    /// Cap on modes per cycle (see [`DEFAULT_MAX_MODES`]).
    pub max_modes: usize,
}

impl Default for ModeConfig {
    fn default() -> Self {
        Self {
            theta: DEFAULT_MODE_THETA,
            max_modes: DEFAULT_MAX_MODES,
        }
    }
}

/// Seed of a mode just distilled from a cycle (still without physics: the entropy trace is anchored in
/// `IMPRINT`, which knows `now` and the resilience). It is the product of the clustering in `DISTILL`.
#[derive(Debug, Clone)]
pub struct ModeSeed {
    /// Centroid of the subgroup (mean of its embeddings).
    pub centroid: Vector,
    /// Dominant lexical label of the mode (mode of the subgroup).
    pub label: String,
    /// `(text, count)` histogram of the mode, sorted by frequency desc.
    pub label_histogram: Vec<(String, usize)>,
    /// Perceptions absorbed by this mode in the cycle.
    pub absorbed: usize,
}

impl ModeSeed {
    /// Consolidates the seed into a live [`Mode`], anchoring its forgetting physics.
    pub fn into_mode(self, halflife: f64, now: Tick) -> Mode {
        Mode {
            trace: EntropyTrace::new(1.0, halflife, now),
            origin: self.centroid.clone(), // birth = the seed's direction; never changes
            centroid: self.centroid,
            label: self.label,
            label_histogram: self.label_histogram,
            absorbed: self.absorbed,
        }
    }
}

/// A consolidated mode in the archetype: a stable behaviour of the subject, with its own life.
#[derive(Debug, Clone)]
pub struct Mode {
    /// Central direction of the mode (accumulated, volume-weighted as it evolves).
    pub centroid: Vector,
    /// **Birth** direction of the mode (its `centroid` the first time it appeared). Fixed — untouched
    /// as it evolves. Basis of `drift`: how much *this behaviour* has changed since it emerged.
    pub origin: Vector,
    /// Dominant lexical label.
    pub label: String,
    /// Accumulated `(text, count)` histogram of the mode.
    pub label_histogram: Vec<(String, usize)>,
    /// Total perceptions this mode represents.
    pub absorbed: usize,
    /// Forgetting physics of the mode: if not revisited, it decays; on recurrence, it is reinforced.
    pub trace: EntropyTrace,
}

impl Mode {
    /// Merges a new seed into this mode (when it resonates with it): moves the centroid toward the new
    /// evidence **volume-weighted** (a 3-event cycle shifts it less than a 3000-event one), accumulates
    /// the histogram and **reinforces** permanence (remembering extends the half-life).
    pub fn merge(&mut self, seed: &ModeSeed, now: Tick) {
        if self.centroid.len() == seed.centroid.len() {
            let w_old = self.absorbed.max(1) as f32;
            let w_new = seed.absorbed.max(1) as f32;
            let total = w_old + w_new;
            for (c, x) in self.centroid.iter_mut().zip(&seed.centroid) {
                *c = (*c * w_old + *x * w_new) / total;
            }
        }
        merge_histograms(&mut self.label_histogram, &seed.label_histogram);
        self.label = self
            .label_histogram
            .first()
            .map(|(t, _)| t.clone())
            .unwrap_or_else(|| self.label.clone());
        self.absorbed += seed.absorbed;
        // Soft consolidation: the recurrent mode gains permanence (like the synapse).
        self.trace.reinforce(now, 0.1);
    }

    /// **Mode drift**: how far its behaviour has shifted since it was born,
    /// `1 − cos(centroid, origin) ∈ [0, 2]`. The identity (`origin`) is fixed; the `centroid` evolves
    /// on recurrence, so high drift = the subject still has this mode but its shape changed (e.g.
    /// "thriller" → "true crime"). This gives a **per-mode trajectory**, not just the global centroid's.
    pub fn drift(&self) -> f32 {
        (1.0 - cosine(&self.centroid, &self.origin)).max(0.0)
    }
}

/// Merges the `src` histogram into `dst` (sums counts per label) and re-sorts it by frequency desc
/// with a deterministic alphabetical tiebreak.
fn merge_histograms(dst: &mut Vec<(String, usize)>, src: &[(String, usize)]) {
    let mut map: HashMap<String, usize> = dst.drain(..).collect();
    for (t, c) in src {
        *map.entry(t.clone()).or_insert(0) += *c;
    }
    let mut merged: Vec<(String, usize)> = map.into_iter().collect();
    merged.sort_by(|(ta, ca), (tb, cb)| cb.cmp(ca).then_with(|| ta.cmp(tb)));
    *dst = merged;
}

/// Internal accumulator of a leader during clustering (unnormalized sum + label frequencies).
struct Leader {
    sum: Vec<f32>,
    count: usize,
    freq: HashMap<String, usize>,
}

/// **Multi-modal DISTILL**: decomposes a set of perceptions into coherent modes.
///
/// Deterministic *leader / DP-means* algorithm: traverses the perceptions in order; assigns each to
/// the nearest leader by cosine if it exceeds `cfg.theta`, or opens a new leader (up to
/// `cfg.max_modes`; once the cap is reached, it assigns to the nearest). Cosine is scale-invariant, so
/// comparing against the leader's **sum** is equivalent to comparing against its centroid — no division
/// in the loop.
///
/// Returns the modes in order of appearance (deterministic). Empty if there are no perceptions.
pub fn cluster_modes(perceptions: &[&Perception], cfg: ModeConfig) -> Vec<ModeSeed> {
    if perceptions.is_empty() {
        return Vec::new();
    }
    let dim = perceptions[0].embedding.len();
    let mut leaders: Vec<Leader> = Vec::new();

    for p in perceptions {
        if p.embedding.len() != dim {
            continue; // robustness: ignore incompatible dimensions instead of corrupting centroids
        }
        // Nearest leader by cosine (against the unnormalized sum — same ranking as the centroid).
        let mut best = f32::NEG_INFINITY;
        let mut best_i: Option<usize> = None;
        for (i, l) in leaders.iter().enumerate() {
            let c = cosine(&p.embedding, &l.sum);
            if c > best {
                best = c;
                best_i = Some(i);
            }
        }
        let target = match best_i {
            Some(i) if best >= cfg.theta => i,
            _ if leaders.len() < cfg.max_modes => {
                leaders.push(Leader {
                    sum: vec![0.0; dim],
                    count: 0,
                    freq: HashMap::new(),
                });
                leaders.len() - 1
            }
            Some(i) => i, // cap reached → to the nearest
            None => unreachable!(
                "no leaders only on the first iteration, covered by the creation branch"
            ),
        };
        let l = &mut leaders[target];
        for (a, x) in l.sum.iter_mut().zip(&p.embedding) {
            *a += *x;
        }
        l.count += 1;
        *l.freq.entry(p.representative_text()).or_insert(0) += 1;
    }

    leaders
        .into_iter()
        .map(|l| {
            let inv = 1.0 / l.count as f32;
            let centroid: Vector = l.sum.iter().map(|x| x * inv).collect();
            let mut hist: Vec<(String, usize)> = l.freq.into_iter().collect();
            hist.sort_by(|(ta, ca), (tb, cb)| cb.cmp(ca).then_with(|| ta.cmp(tb)));
            let label = hist.first().map(|(t, _)| t.clone()).unwrap_or_default();
            ModeSeed {
                centroid,
                label,
                label_histogram: hist,
                absorbed: l.count,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(act: &str, e: Vec<f32>) -> Perception {
        Perception::new("u", e, 1.0, 3600.0, 0.0).with_trait("act", act)
    }

    #[test]
    fn unimodal_set_yields_one_mode() {
        // A single behaviour → one mode (reproduces the old single-centroid case).
        let ps = [
            p("trail", vec![1.0, 0.0]),
            p("trail", vec![0.99, 0.01]),
            p("trail", vec![1.0, 0.02]),
        ];
        let refs: Vec<&Perception> = ps.iter().collect();
        let modes = cluster_modes(&refs, ModeConfig::default());
        assert_eq!(modes.len(), 1);
        assert_eq!(modes[0].absorbed, 3);
        assert_eq!(modes[0].label, "trail");
    }

    #[test]
    fn three_distinct_behaviors_yield_three_modes() {
        // Three orthogonal behaviours → three modes, not a mean in the centre.
        let mut ps = Vec::new();
        for _ in 0..5 {
            ps.push(p("noir", vec![1.0, 0.0, 0.0]));
            ps.push(p("docs", vec![0.0, 1.0, 0.0]));
            ps.push(p("scifi", vec![0.0, 0.0, 1.0]));
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let modes = cluster_modes(&refs, ModeConfig::default());
        assert_eq!(
            modes.len(),
            3,
            "three coherent modes, not a noisy average"
        );
        let labels: Vec<&str> = modes.iter().map(|m| m.label.as_str()).collect();
        assert!(labels.contains(&"noir") && labels.contains(&"docs") && labels.contains(&"scifi"));
        // Each mode points to its direction, not the mean (which would be ~(0.33,0.33,0.33)).
        for m in &modes {
            let max = m.centroid.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                max > 0.9,
                "the mode's centroid is sharp, not the mean: {:?}",
                m.centroid
            );
        }
    }

    #[test]
    fn merge_blends_by_volume_and_reinforces() {
        let halflife = 3600.0;
        let mut mode = ModeSeed {
            centroid: vec![1.0, 0.0],
            label: "a".into(),
            label_histogram: vec![("a".into(), 100)],
            absorbed: 100,
        }
        .into_mode(halflife, 0.0);
        let r0 = mode.trace.reinforcement;
        mode.merge(
            &ModeSeed {
                centroid: vec![0.0, 1.0],
                label: "b".into(),
                label_histogram: vec![("b".into(), 50)],
                absorbed: 50,
            },
            1.0,
        );
        // (100·1, 50·1)/150 = (0.667, 0.333) — volume-weighted, not (0.5, 0.5).
        assert!((mode.centroid[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((mode.centroid[1] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(mode.absorbed, 150);
        assert!(
            mode.trace.reinforcement > r0,
            "merging a recurrent mode reinforces it"
        );
    }

    #[test]
    fn mode_drift_grows_as_behavior_shifts_but_origin_is_fixed() {
        let mut mode = ModeSeed {
            centroid: vec![1.0, 0.0],
            label: "a".into(),
            label_histogram: vec![("a".into(), 100)],
            absorbed: 100,
        }
        .into_mode(3600.0, 0.0);
        // Newborn: centroid == origin → no drift.
        assert!(mode.drift() < 1e-6, "newborn mode has not drifted");

        // The behaviour shifts (new evidence moves the centroid), the origin does NOT change.
        mode.merge(
            &ModeSeed {
                centroid: vec![0.0, 1.0],
                label: "a".into(),
                label_histogram: vec![("a".into(), 100)],
                absorbed: 100,
            },
            1.0,
        );
        // (100·[1,0] + 100·[0,1])/200 = [0.5,0.5]; cos([0.5,0.5],[1,0]) ≈ 0.707 → drift ≈ 0.293.
        assert!(
            mode.drift() > 0.25,
            "the mode drifted from its origin: {}",
            mode.drift()
        );
        assert_eq!(
            mode.origin,
            vec![1.0, 0.0],
            "the origin is still the birth direction"
        );
    }
}
