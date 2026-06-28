//! Archetype layer — long-term memory (`IMPRINT`).
//!
//! Intention Vectors consistent across cycles consolidate into an `Archetype`: the subject's essence
//! in a few dense vectors. **Evolution anchor**: not immortal — still subject to the physics, but with
//! high resilience. Embedded storage with linear Flat search (cosine); the ANN index (HNSW) arrives
//! to scale.

use crate::entropy::{EntropyTrace, Tick};
use crate::modes::{Mode, DEFAULT_MODE_THETA};
use crate::synthesis::IntentionVector;
use crate::vector::{cosine, Vector};

/// Resilience of an archetype to forgetting (modulates the base half-life of the `IMPRINT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resilience {
    Low,
    Medium,
    High,
}

/// Base half-lives (seconds) of each resilience level. **Declared physics** (not magic constants
/// buried in a `match`): a `Low` archetype lasts ~1 month, `Medium` ~6 months, `High` ~2 years before
/// its weight halves without reinforcement (consolidation by evocation extends them). Public and
/// calibratable, in the same vocabulary as the rest of the engine's thresholds (`DEFAULT_THETA_FADE`…).
pub const HALFLIFE_LOW_SECS: f64 = 30.0 * 86_400.0;
pub const HALFLIFE_MEDIUM_SECS: f64 = 180.0 * 86_400.0;
pub const HALFLIFE_HIGH_SECS: f64 = 720.0 * 86_400.0;

impl Resilience {
    /// Base half-life (seconds) of the level. See [`HALFLIFE_LOW_SECS`] / [`HALFLIFE_MEDIUM_SECS`] /
    /// [`HALFLIFE_HIGH_SECS`].
    pub fn halflife(self) -> f64 {
        match self {
            Resilience::Low => HALFLIFE_LOW_SECS,
            Resilience::Medium => HALFLIFE_MEDIUM_SECS,
            Resilience::High => HALFLIFE_HIGH_SECS,
        }
    }
}

/// A milestone in the evolutionary arc: the subject's direction in a given dream cycle.
#[derive(Debug, Clone)]
pub struct ArcMilestone {
    pub at: Tick,
    pub direction: Vector,
    pub absorbed: usize,
    /// Dominant lexical label of that cycle (what occupied the subject back then).
    pub label: String,
    /// `(text, count)` histogram of the cycle: the mix of behaviours, not just the dominant one.
    /// Basis of the per-domain trajectories in `evoke` (answering "did X come back?").
    pub label_histogram: Vec<(String, usize)>,
}

/// The consolidated essence of a subject.
#[derive(Debug, Clone)]
pub struct Archetype {
    pub subject: String,
    /// Stable core: accumulated central direction of behaviour (GLOBAL mean). Kept as the arc origin
    /// and backwards-compatible resonance; the rich representation lives in `modes`.
    pub core: Vector,
    /// The subject's **modes**: distinct coherent behaviours, each with its own physics. The global
    /// mean (`core`) collapsed disparate behaviours into noise; the modes separate them, and resonance
    /// recovers the relevant mode, not the average. See [`crate::modes`].
    pub modes: Vec<Mode>,
    /// Retained novelty vectors (pattern breaks that still resonate).
    pub anomalies: Vec<Vector>,
    /// Lexical labels of the anomalies, aligned with `anomalies`.
    pub anomaly_labels: Vec<String>,
    /// Lexical label of the **current** dominant behaviour (last consolidated cycle).
    pub core_label: String,
    /// Total perceptions this essence represents (denominator of the compression ratio).
    pub represented: usize,
    /// Arc milestones: one direction per dream cycle. The subject's trajectory over time.
    pub arc: Vec<ArcMilestone>,
    /// Forgetting physics of the archetype itself (evolution anchor).
    pub trace: EntropyTrace,
}

impl Archetype {
    /// `IMPRINT`: consolidates an Intention Vector into a new archetype.
    pub fn imprint(iv: &IntentionVector, resilience: Resilience, now: Tick) -> Self {
        let arc = vec![ArcMilestone {
            at: now,
            direction: iv.centroid.clone(),
            absorbed: iv.absorbed,
            label: iv.core_label.clone(),
            label_histogram: iv.label_histogram.clone(),
        }];
        let halflife = resilience.halflife();
        let modes = iv
            .modes
            .iter()
            .cloned()
            .map(|s| s.into_mode(halflife, now))
            .collect();
        Self {
            subject: iv.subject.clone(),
            core: iv.centroid.clone(),
            modes,
            anomalies: iv.anomalies.clone(),
            anomaly_labels: iv.anomaly_labels.clone(),
            core_label: iv.core_label.clone(),
            represented: iv.absorbed,
            arc,
            trace: EntropyTrace::new(1.0, resilience.halflife(), now),
        }
    }

    /// Evolves the archetype: absorbs a new Intention Vector by moving the core toward the new
    /// direction, reinforcing its permanence, and recording a milestone in the arc. The essence
    /// *evolves*, it is not replaced.
    pub fn evolve(&mut self, iv: &IntentionVector, now: Tick) {
        // **Volume-weighted** blend: the core shifts toward the new direction in proportion to the
        // evidence backing it. A 3-event cycle should not displace identity as much as a 30,000-
        // event one.
        if self.core.len() == iv.centroid.len() {
            let w_old = self.represented.max(1) as f32;
            let w_new = iv.absorbed.max(1) as f32;
            let total = w_old + w_new;
            for (c, x) in self.core.iter_mut().zip(&iv.centroid) {
                *c = (*c * w_old + *x * w_new) / total;
            }
        }
        // Modes: each new mode is merged into the existing mode with highest resonance (≥ θ), or
        // born as its own mode (with the archetype's already-consolidated half-life). Recurring
        // behaviour is reinforced; new behaviour is added without contaminating the others.
        let halflife = crate::entropy::halflife_from_lambda(self.trace.lambda);
        for seed in &iv.modes {
            let mut best = DEFAULT_MODE_THETA;
            let mut best_i: Option<usize> = None;
            for (i, m) in self.modes.iter().enumerate() {
                let c = cosine(&m.centroid, &seed.centroid);
                if c >= best {
                    best = c;
                    best_i = Some(i);
                }
            }
            match best_i {
                Some(i) => self.modes[i].merge(seed, now),
                None => self.modes.push(seed.clone().into_mode(halflife, now)),
            }
        }

        self.anomalies.extend(iv.anomalies.iter().cloned());
        self.anomaly_labels
            .extend(iv.anomaly_labels.iter().cloned());
        // Current dominant interest is that of the last consolidated cycle.
        self.core_label = iv.core_label.clone();
        self.represented += iv.absorbed;
        self.arc.push(ArcMilestone {
            at: now,
            direction: iv.centroid.clone(),
            absorbed: iv.absorbed,
            label: iv.core_label.clone(),
            label_histogram: iv.label_histogram.clone(),
        });
        // Soft consolidation: recalling/evolving extends the half-life.
        self.trace.reinforce(now, 0.1);
    }

    /// Cosine resonance between a query and the subject. With modes, returns the **maximum** over
    /// all modes (the query recovers the behaviour it truly concerns, not the global mean that
    /// dilutes the signal). Without modes (legacy archetype), falls back to the global core.
    /// Basis of `EVOKE`.
    pub fn resonance(&self, query: &[f32]) -> f32 {
        if self.modes.is_empty() {
            return cosine(&self.core, query);
        }
        self.modes
            .iter()
            .map(|m| cosine(&m.centroid, query))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Label of the **mode with highest resonance** for a query (basis of `EVOKE … RESONATING WITH`):
    /// evocation focuses on the aspect of the subject that concerns the queried trait, not its
    /// global dominant behaviour. `None` if the archetype has no modes (legacy).
    pub fn resonant_mode_label(&self, query: &[f32]) -> Option<String> {
        self.modes
            .iter()
            .map(|m| (cosine(&m.centroid, query), &m.label))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, label)| label.clone())
    }

    /// Real `IMPRINT`: **consolidates/anchors** an already existing essence. Does not create or evolve
    /// (that is `DISTILL`/`evolve`): reinforces the physics of the archetype **and each mode**
    /// (Δt→0, λ reduced by `consolidation` → longer half-life), so the essence gains permanence
    /// against forgetting. `consolidation ∈ [0, 1)`: how much λ is reduced (0 = only resets Δt and
    /// adds reinforcement).
    pub fn consolidate(&mut self, now: Tick, consolidation: f64) {
        self.trace.reinforce(now, consolidation);
        for m in &mut self.modes {
            m.trace.reinforce(now, consolidation);
        }
    }
}

/// Long-term memory: the set of live archetypes. Linear Flat search (exact; ANN is L3).
#[derive(Debug, Default)]
pub struct ArchetypeStore {
    archetypes: Vec<Archetype>,
}

impl ArchetypeStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Evolutionary `IMPRINT`: if an archetype exists for the subject, evolves it; otherwise creates one.
    pub fn imprint(&mut self, iv: &IntentionVector, resilience: Resilience, now: Tick) {
        if let Some(a) = self.archetypes.iter_mut().find(|a| a.subject == iv.subject) {
            a.evolve(iv, now);
        } else {
            self.archetypes
                .push(Archetype::imprint(iv, resilience, now));
        }
    }

    /// Returns the archetype for a subject, if it exists.
    pub fn get(&self, subject: &str) -> Option<&Archetype> {
        self.archetypes.iter().find(|a| a.subject == subject)
    }

    /// Iterates over archetypes (for snapshots / persistence).
    pub fn iter(&self) -> impl Iterator<Item = &Archetype> {
        self.archetypes.iter()
    }

    /// Inserts an already-built archetype (e.g. restored from disk). Does not merge: assumes the
    /// subject is not yet present. Use when rehydrating an empty store.
    pub fn insert(&mut self, archetype: Archetype) {
        self.archetypes.push(archetype);
    }

    /// `IMPRINT`: consolidates (anchors) a subject's archetype if it exists. Returns `false` if none
    /// exists — you cannot imprint what has not been distilled. See [`Archetype::consolidate`].
    pub fn consolidate(&mut self, subject: &str, now: Tick, consolidation: f64) -> bool {
        match self.archetypes.iter_mut().find(|a| a.subject == subject) {
            Some(a) => {
                a.consolidate(now, consolidation);
                true
            }
            None => false,
        }
    }

    /// Linear Flat search by **life-weighted resonance**: the `k` archetypes with highest score for
    /// the query. The score is NOT raw cosine, but `score = relevance · life`:
    ///
    /// ```text
    /// score = max(0, resonance(query)) · weight(now)
    /// ```
    ///
    /// where `weight(now) = salience · e^(−λΔt) · (1 + reinforcement)` integrates **recency** (decay),
    /// **importance** (salience), and **reinforcement**. This is the engine's native physics used for
    /// ranking — a highly relevant but faded memory ranks below an equally relevant but live one.
    /// (Multiplicative form of Generative Agents retrieval, without magic α/β/γ coefficients to tune.)
    pub fn resonate(&self, query: &[f32], k: usize, now: Tick, theta_fade: f64) -> Vec<&Archetype> {
        let mut scored: Vec<(f64, &Archetype)> = self
            .archetypes
            .iter()
            .filter_map(|a| {
                // `weight()` includes `e^x`: evaluated **only once** per archetype (was 2×).
                let life = a.trace.weight(now);
                if life < theta_fade {
                    return None; // faded archetype
                }
                let relevance = a.resonance(query).max(0.0) as f64;
                Some((relevance * life, a))
            })
            .collect();
        scored.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, a)| a).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::IntentionVector;

    fn iv(subject: &str, c: Vec<f32>, absorbed: usize) -> IntentionVector {
        IntentionVector {
            subject: subject.to_string(),
            centroid: c,
            anomalies: vec![],
            core_label: format!("{subject}-core"),
            anomaly_labels: vec![],
            absorbed,
            redundant: 0,
            label_histogram: vec![(format!("{subject}-core"), absorbed)],
            modes: vec![],
        }
    }

    #[test]
    fn imprint_creates_then_evolves() {
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("user:X", vec![1.0, 0.0], 100), Resilience::High, 0.0);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("user:X").unwrap().represented, 100);

        // Second cycle: same subject evolves, is not duplicated.
        store.imprint(&iv("user:X", vec![0.0, 1.0], 50), Resilience::High, 3600.0);
        assert_eq!(store.len(), 1);
        let a = store.get("user:X").unwrap();
        assert_eq!(a.represented, 150);
        // Core shifts **volume-weighted**: 100 old events ([1,0]) + 50 new ([0,1])
        // ⇒ (100·1, 50·1)/150 = (0.667, 0.333), not (0.5, 0.5).
        assert!(
            (a.core[0] - 2.0 / 3.0).abs() < 1e-6,
            "core[0] = {}",
            a.core[0]
        );
        assert!(
            (a.core[1] - 1.0 / 3.0).abs() < 1e-6,
            "core[1] = {}",
            a.core[1]
        );
    }

    #[test]
    fn evolve_weights_by_evidence_volume() {
        // A tiny break (1 event) barely moves a consolidated identity (10,000 events).
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("u", vec![1.0, 0.0], 10_000), Resilience::High, 0.0);
        store.imprint(&iv("u", vec![0.0, 1.0], 1), Resilience::High, 1.0);
        let core = &store.get("u").unwrap().core;
        assert!(core[0] > 0.999, "identity barely moved: {core:?}");
    }

    #[test]
    fn resonate_ranks_by_cosine() {
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("user:A", vec![1.0, 0.0], 10), Resilience::High, 0.0);
        store.imprint(&iv("user:B", vec![0.0, 1.0], 10), Resilience::High, 0.0);

        let top = store.resonate(&[0.9, 0.1], 1, 0.0, crate::entropy::DEFAULT_THETA_FADE);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].subject, "user:A");
    }

    #[test]
    fn resonate_weights_relevance_by_life_l2() {
        // Two subjects with EQUAL relevance (same direction). Only thing distinguishing them is LIFE:
        // one was just touched, the other has been decaying for a half-life. Physics must rank the
        // fresh one first even though cosine is identical — raw cosine could not tell them apart.
        let half = Resilience::Low.halflife(); // 30 days
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("user:stale", vec![1.0, 0.0], 10), Resilience::Low, 0.0);
        store.imprint(&iv("user:fresh", vec![1.0, 0.0], 10), Resilience::Low, half);

        let ranked = store.resonate(&[1.0, 0.0], 2, half, crate::entropy::DEFAULT_THETA_FADE);
        assert_eq!(ranked.len(), 2, "both still alive");
        assert_eq!(
            ranked[0].subject, "user:fresh",
            "equal relevance: freshest ranks first"
        );
        assert_eq!(ranked[1].subject, "user:stale");
    }

    #[test]
    fn high_resilience_outlives_low() {
        assert!(Resilience::High.halflife() > Resilience::Low.halflife());
    }

    #[test]
    fn multimodal_resonance_recovers_the_right_mode_where_centroid_is_blind() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};

        // A subject with TWO opposite behaviours: the global mean is the NULL vector → the single
        // centroid is blind (resonance 0 for any query). This is the pathological case that destroyed
        // signal in multi-modal data. With modes, each behaviour is preserved distinctly.
        let mut ps = Vec::new();
        for _ in 0..50 {
            ps.push(
                Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "left"),
            );
            ps.push(
                Perception::new("u", vec![-1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "right"),
            );
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        assert_eq!(
            iv.modes.len(),
            2,
            "two opposite behaviours → two modes"
        );

        let mut store = ArchetypeStore::new();
        store.imprint(&iv, Resilience::High, 0.0);
        let a = store.get("u").unwrap();

        // Global core is ~null: the single-centroid path CANNOT retrieve anything.
        assert!(
            cosine(&a.core, &[1.0, 0.0]).abs() < 1e-3,
            "single centroid is blind: {:?}",
            a.core
        );
        // Multi-modal resonance recovers the behaviour actually relevant to the query.
        assert!(
            (a.resonance(&[1.0, 0.0]) - 1.0).abs() < 1e-3,
            "correct mode resonates fully"
        );
        assert!(
            (a.resonance(&[-1.0, 0.0]) - 1.0).abs() < 1e-3,
            "other mode also resonates fully for its query"
        );
    }

    #[test]
    fn resonant_mode_label_picks_the_aspect_that_matches_the_query() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};
        // Two orthogonal labelled behaviours → two modes.
        let mut ps = Vec::new();
        for _ in 0..6 {
            ps.push(
                Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "noir"),
            );
            ps.push(
                Perception::new("u", vec![0.0, 1.0], 1.0, 3600.0, 0.0).with_trait("act", "docs"),
            );
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        let mut store = ArchetypeStore::new();
        store.imprint(&iv, Resilience::High, 0.0);
        let a = store.get("u").unwrap();
        // Each query recovers the label of the mode it concerns (basis of RESONATING WITH).
        assert_eq!(a.resonant_mode_label(&[1.0, 0.0]).as_deref(), Some("noir"));
        assert_eq!(a.resonant_mode_label(&[0.0, 1.0]).as_deref(), Some("docs"));
    }

    #[test]
    fn consolidate_anchors_archetype_and_modes() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};
        let ps: Vec<Perception> = (0..4)
            .map(|_| Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "x"))
            .collect();
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        let mut store = ArchetypeStore::new();
        store.imprint(&iv, Resilience::High, 0.0);
        let (r0, lambda0, mr0) = {
            let a = store.get("u").unwrap();
            (
                a.trace.reinforcement,
                a.trace.lambda,
                a.modes[0].trace.reinforcement,
            )
        };
        // Real IMPRINT consolidates: more reinforcement, lower λ (longer half-life), modes anchored too.
        assert!(store.consolidate("u", 10.0, 0.2));
        let a = store.get("u").unwrap();
        assert!(a.trace.reinforcement > r0, "archetype gains reinforcement");
        assert!(a.trace.lambda < lambda0, "λ drops → permanence gained");
        assert!(
            a.modes[0].trace.reinforcement > mr0,
            "modes are also anchored"
        );
        // Cannot imprint what has not been distilled.
        assert!(!store.consolidate("ghost", 10.0, 0.2));
    }
}
