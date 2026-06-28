//! The Cognitive Runtime — the organism that "breathes".
//!
//! Orchestrates three layers: perception (short-term) → synthesis (sleep) → archetype (long-term),
//! with FADE as the semantic garbage-collector sweep.
//!
//! The "breathing" is synchronous and driven by logical ticks (deterministic, testable, offline);
//! the `letheo-async` crate mounts it on Tokio to run asynchronously without blocking perception
//! while the engine sleeps. The physics is **lazy** in either case:
//! `breathe()` is the only point where weights are recalculated in bulk.

use crate::archetype::{ArchetypeStore, Resilience};
use crate::entropy::{Tick, DEFAULT_THETA_FADE};
use crate::evoke::{evoke, evoke_unified, CompressedContext, EvokeRequest, UnifiedContext};
use crate::factstore::{FactStore, RecalledFact, Remember};
use crate::perception::{Perception, PerceptionBuffer};
use crate::synthesis::{distill, DistillConfig};
use crate::vector::Vector;

/// Runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub theta_fade: f64,
    pub distill: DistillConfig,
    pub resilience: Resilience,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            theta_fade: DEFAULT_THETA_FADE,
            distill: DistillConfig::default(),
            resilience: Resilience::High,
        }
    }
}

/// Report from a breath cycle (for observability / sandbox).
#[derive(Debug, Default, Clone)]
pub struct BreathReport {
    pub distilled_subjects: usize,
    pub perceptions_absorbed: usize,
    pub faded: usize,
}

/// The cognitive runtime: perceives, dreams, evokes, fades. Holds **both layers** under the same
/// physics: the semantic (`long_term`, archetypes/modes — layer-2) and the episodic (`facts`, verbatim
/// facts — layer-1). A single evocation unifies them (see [`evoke_unified`](Self::evoke_unified)).
pub struct CognitiveRuntime {
    short_term: PerceptionBuffer,
    long_term: ArchetypeStore,
    facts: FactStore,
    cfg: RuntimeConfig,
}

impl CognitiveRuntime {
    pub fn new(cfg: RuntimeConfig) -> Self {
        Self {
            short_term: PerceptionBuffer::new(),
            long_term: ArchetypeStore::new(),
            facts: FactStore::new(),
            cfg,
        }
    }

    /// `PERCEIVE`: assimilates a raw stimulus into short-term memory.
    pub fn perceive(&mut self, p: Perception) {
        self.short_term.perceive(p);
    }

    /// One "sleep" cycle: for each subject with live perceptions, `DISTILL` → `IMPRINT`, then
    /// `FADE` sweeps the already-absorbed noise. This is the only point of bulk weight recalculation.
    pub fn breathe(&mut self, subjects: &[&str], now: Tick) -> BreathReport {
        let mut report = BreathReport::default();

        for &subject in subjects {
            let alive: Vec<&Perception> = self
                .short_term
                .alive_for(subject, now, self.cfg.theta_fade)
                .collect();
            if alive.is_empty() {
                continue;
            }
            if let Some(iv) = distill(subject, &alive, self.cfg.distill) {
                report.perceptions_absorbed += iv.absorbed;
                self.long_term.imprint(&iv, self.cfg.resilience, now);
                report.distilled_subjects += 1;
            }
        }

        // FADE: noise whose vote already lives in the archetype fades away.
        report.faded = self.short_term.fade_swept(now, self.cfg.theta_fade);
        report
    }

    /// Like [`breathe`](Self::breathe) but only distils perceptions satisfying the `keep` predicate
    /// (`WHERE` clause of `DISTILL`). The subsequent `FADE` sweep remains global.
    pub fn breathe_where(
        &mut self,
        subjects: &[&str],
        now: Tick,
        keep: impl Fn(&Perception) -> bool,
    ) -> BreathReport {
        let mut report = BreathReport::default();

        for &subject in subjects {
            let alive: Vec<&Perception> = self
                .short_term
                .alive_for_where(subject, now, self.cfg.theta_fade, &keep)
                .collect();
            if alive.is_empty() {
                continue;
            }
            if let Some(iv) = distill(subject, &alive, self.cfg.distill) {
                report.perceptions_absorbed += iv.absorbed;
                self.long_term.imprint(&iv, self.cfg.resilience, now);
                report.distilled_subjects += 1;
            }
        }

        report.faded = self.short_term.fade_swept(now, self.cfg.theta_fade);
        report
    }

    /// `FADE … WHERE`: explicit sweep of perceptions satisfying the predicate.
    pub fn fade_where(&mut self, drop_if: impl Fn(&Perception) -> bool) -> usize {
        self.short_term.fade_swept_where(drop_if)
    }

    /// `EVOKE`: resolves the essence of a subject within the token budget (layer-2 only).
    pub fn evoke(&self, req: &EvokeRequest, now: Tick) -> Option<CompressedContext> {
        evoke(&self.long_term, req, now)
    }

    /// Stores an **episodic fact** (layer-1): verbatim content + embedding, under forgetting physics.
    /// Semantic dedup per subject: a repeated fact is reinforced, not duplicated.
    #[allow(clippy::too_many_arguments)]
    pub fn remember(
        &mut self,
        subject: impl Into<String>,
        text: impl Into<String>,
        embedding: Vector,
        provenance: impl Into<String>,
        salience: f64,
        halflife: f64,
        now: Tick,
    ) -> Remember {
        self.facts.remember(
            subject, text, embedding, provenance, salience, halflife, now,
        )
    }

    /// `RECALL` (layer-1): retrieves the most relevant exact facts for a subject by physics
    /// (`relevance · life`) and **reinforces them** (spaced repetition: recalling resets their decay).
    pub fn recall(
        &mut self,
        subject: &str,
        query: &[f32],
        k: usize,
        now: Tick,
    ) -> Vec<RecalledFact> {
        self.facts
            .recall(subject, query, k, now, self.cfg.theta_fade)
    }

    /// **Unified `EVOKE`**: a single evocation that answers character (layer-2) **and** nominal
    /// (layer-1) under ONE budget. `fact_budget` is the portion reserved for facts; `fact_cost` is
    /// the real tokenizer injected. Read-only (see [`evoke_unified`]).
    pub fn evoke_unified(
        &self,
        req: &EvokeRequest,
        query: &[f32],
        fact_budget: usize,
        now: Tick,
        fact_cost: impl Fn(&str) -> usize,
    ) -> UnifiedContext {
        evoke_unified(
            &self.long_term,
            &self.facts,
            req,
            query,
            fact_budget,
            now,
            fact_cost,
        )
    }

    /// `IMPRINT`: **consolidates (anchors)** a subject's archetype — reinforces its physics and its
    /// modes' physics for permanence. Returns `false` if there is no archetype (cannot imprint what
    /// has not been distilled). See [`crate::Archetype::consolidate`].
    pub fn consolidate(&mut self, subject: &str, now: Tick, consolidation: f64) -> bool {
        self.long_term.consolidate(subject, now, consolidation)
    }

    /// **Reflection** (L8): higher-order insights about the subject's trajectory — dominant transitions
    /// and revivals — absent from any individual event. Empty if there is no archetype.
    /// See [`crate::reflection::reflect`].
    pub fn reflect(&self, subject: &str) -> Vec<crate::reflection::Insight> {
        self.long_term
            .get(subject)
            .map(|a| crate::reflection::reflect(&a.arc))
            .unwrap_or_default()
    }

    /// **Reflective sleep** (L8): reflects over the subject's arc and **materialises** insights as
    /// high-salience facts in layer-1 (with embeddings derived from archetype geometry, no provider).
    /// This makes distilled arc wisdom retrievable via `RECALL`. Returns how many insights were stored.
    /// Intended to be called in the sleep cycle after `breathe`.
    pub fn dream_reflect(&mut self, subject: &str, now: Tick) -> usize {
        // Compute materialised insights with a borrowed archetype, then drop the borrow
        // before writing into layer-1 (disjoint fields, disjoint borrows).
        let materials: Vec<(String, crate::vector::Vector)> = match self.long_term.get(subject) {
            Some(a) => crate::reflection::reflect(&a.arc)
                .iter()
                .filter_map(|ins| crate::reflection::materialize(a, ins))
                .collect(),
            None => return 0,
        };
        let halflife = 90.0 * 86_400.0; // 90 days: insights are durable but not immortal.
        let n = materials.len();
        for (text, embedding) in materials {
            self.facts.remember(
                subject,
                text,
                embedding,
                "reflection",
                crate::reflection::DEFAULT_INSIGHT_SALIENCE,
                halflife,
                now,
            );
        }
        n
    }

    /// `FADE` of the episodic layer: sweeps facts whose life dropped below θ_fade. Returns how many.
    pub fn fade_facts(&mut self, now: Tick) -> usize {
        self.facts.fade(now, self.cfg.theta_fade)
    }

    /// Read-only access to episodic memory (for persistence with `letheo-persist`).
    pub fn facts(&self) -> &FactStore {
        &self.facts
    }

    /// Mutable access to episodic memory (for rehydrating from disk).
    pub fn facts_mut(&mut self) -> &mut FactStore {
        &mut self.facts
    }

    /// Explicit `FADE`: semantic GC sweep without a full sleep cycle. Useful when an MQL program
    /// wants to express forgetting as an act without triggering `DISTILL`.
    pub fn fade_only(&mut self, now: Tick, theta: f64) -> usize {
        self.short_term.fade_swept(now, theta)
    }

    /// Read-only access to long-term memory (for persistence).
    pub fn long_term(&self) -> &ArchetypeStore {
        &self.long_term
    }

    /// Mutable access to long-term memory (for rehydrating from disk).
    pub fn long_term_mut(&mut self) -> &mut ArchetypeStore {
        &mut self.long_term
    }

    pub fn short_term_len(&self) -> usize {
        self.short_term.len()
    }

    pub fn long_term_len(&self) -> usize {
        self.long_term.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HF: f64 = 3600.0;

    fn perception(subject: &str, e: Vec<f32>, salience: f64) -> Perception {
        Perception::new(subject, e, salience, HF, 0.0)
    }

    #[test]
    fn full_cycle_perceive_breathe_evoke() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());

        // 1000 nearly identical perceptions (a habit) + a few anomalous ones.
        for _ in 0..1000 {
            rt.perceive(perception("user:Xolotl", vec![1.0, 0.0], 1.0));
        }
        for _ in 0..3 {
            rt.perceive(perception("user:Xolotl", vec![0.0, 1.0], 1.0));
        }
        assert_eq!(rt.short_term_len(), 1003);

        // The runtime sleeps.
        let report = rt.breathe(&["user:Xolotl"], 0.0);
        assert_eq!(report.distilled_subjects, 1);
        assert_eq!(report.perceptions_absorbed, 1003);
        assert_eq!(rt.long_term_len(), 1, "one consolidated essence");

        // EVOKE returns ultra-compressed context within the budget.
        let req = EvokeRequest::new("user:Xolotl", 800);
        let ctx = rt.evoke(&req, 0.0).unwrap();
        assert_eq!(ctx.represented, 1003);
        assert!(ctx.token_estimate <= 800);
        assert!(ctx.compression_ratio() > 100.0);
    }

    #[test]
    fn unified_runtime_evoke_spans_both_layers() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        // Layer-2: perceive a habit and sleep → character gist.
        for _ in 0..100 {
            rt.perceive(Perception::new(
                "user:X",
                vec![1.0, 0.0],
                1.0,
                86_400.0,
                0.0,
            ));
        }
        rt.breathe(&["user:X"], 0.0);
        // Layer-1: store an exact fact (the nominal).
        rt.remember(
            "user:X",
            "prefers window seat",
            vec![0.0, 1.0],
            "agent",
            1.0,
            86_400.0,
            0.0,
        );

        let req = EvokeRequest::new("user:X", 800);
        let u = rt.evoke_unified(
            &req,
            &[0.0, 1.0],
            100,
            0.0,
            crate::evoke::approx_token_count,
        );
        assert!(u.gist.is_some(), "character from layer-2");
        assert_eq!(u.facts.len(), 1, "nominal from layer-1");
        assert_eq!(u.facts[0].text, "prefers window seat");
        assert!(u.total_tokens <= 800);
    }

    #[test]
    fn runtime_reflect_surfaces_arc_transition() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        // Two sleep cycles with different behaviours → an arc with a trail→yoga transition.
        // Trail decays fast (halflife 1s) so it does not bleed into the second cycle (see D14).
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![1.0, 0.0], 1.0, 1.0, 0.0).with_trait("act", "trail"),
            );
        }
        rt.breathe(&["u"], 0.0);
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![0.0, 1.0], 1.0, 86_400.0, 0.0).with_trait("act", "yoga"),
            );
        }
        rt.breathe(&["u"], 100.0);

        let insights = rt.reflect("u");
        assert!(
            insights.iter().any(
                |i| matches!(i, crate::reflection::Insight::Transition { from, to, .. }
                if from == "trail" && to == "yoga")
            ),
            "reflection synthesises the arc transition: {insights:?}"
        );
        assert!(
            rt.reflect("ghost").is_empty(),
            "no archetype, no insights"
        );
    }

    #[test]
    fn dream_reflect_materializes_insights_as_recallable_facts() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![1.0, 0.0], 1.0, 1.0, 0.0).with_trait("act", "trail"),
            );
        }
        rt.breathe(&["u"], 0.0);
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![0.0, 1.0], 1.0, 86_400.0, 0.0).with_trait("act", "yoga"),
            );
        }
        rt.breathe(&["u"], 100.0);

        // Reflective sleep materialises the transition as a high-salience fact…
        let stored = rt.dream_reflect("u", 100.0);
        assert_eq!(stored, 1, "the trail→yoga transition was stored as a fact");
        // …retrievable by resonance with the target behaviour (layer-1).
        let hits = rt.recall("u", &[0.0, 1.0], 1, 100.0);
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].text.contains("trail → yoga"),
            "the insight is retrieved: {}",
            hits[0].text
        );
    }

    #[test]
    fn noise_fades_after_breathing() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        rt.perceive(perception("user:X", vec![1.0, 0.0], 0.2)); // weak noise
                                                                // We sleep much later: the noise has already dropped below θ_fade.
        let report = rt.breathe(&["user:X"], HF * 5.0);
        assert_eq!(report.faded, 1);
        assert_eq!(rt.short_term_len(), 0);
    }
}
