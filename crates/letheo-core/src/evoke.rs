//! `EVOKE` — semantic resonance with token budget.
//!
//! Does not replay history: resonates with archetypes and reconstructs the essence within a
//! token budget, returning an **ultra-compressed context block** (not a list of events).

use crate::archetype::{Archetype, ArchetypeStore};
use crate::entropy::{Tick, DEFAULT_THETA_FADE};
use crate::factstore::{FactStore, RecalledFact};
use crate::vector::cosine;

/// How much detail of the evolutionary arc to return. Maps MQL clauses `RESOLUTION`/`PROJECTING`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArcDetail {
    /// Full arc, trimmed to budget (`RESOLUTION arc` / `PROJECTING trajectory`).
    #[default]
    Full,
    /// Only a few key milestones (`RESOLUTION summary`).
    Summary,
    /// No arc: current state only (`RESOLUTION point` / `PROJECTING snapshot`).
    None,
}

/// Parameters for an evocation.
#[derive(Debug, Clone)]
pub struct EvokeRequest {
    pub subject: String,
    /// Token budget: the returned context must not exceed it.
    pub token_budget: usize,
    /// Only arc milestones with `at ≥ since` (`None` = entire arc). Maps `ACROSS span`.
    pub since: Option<Tick>,
    /// Arc detail to return. Maps `RESOLUTION`/`PROJECTING`.
    pub arc_detail: ArcDetail,
    /// Token cost of a rendered dense vector. **Declared default**
    /// ([`DEFAULT_TOKENS_PER_VECTOR`]), overridable with the real mean measured by the tokenizer
    /// (tiktoken) in the orchestration layer: the 24 is no longer baked into the algorithm (debt #1 of
    /// TRUTH 100% — the cost is measured or injected, never invented).
    pub tokens_per_vector: usize,
}

impl EvokeRequest {
    /// Evocation with default values (full arc, no time window).
    pub fn new(subject: impl Into<String>, token_budget: usize) -> Self {
        Self {
            subject: subject.into(),
            token_budget,
            since: None,
            arc_detail: ArcDetail::Full,
            tokens_per_vector: DEFAULT_TOKENS_PER_VECTOR,
        }
    }
}

/// The result of `EVOKE`: dense context and its compression metric. The orchestration layer
/// converts it to prose for the LLM; the core delivers the structure.
#[derive(Debug, Clone)]
pub struct CompressedContext {
    pub subject: String,
    /// Number of perceptions this essence represents (numerator of the ratio).
    pub represented: usize,
    /// Number of vectors returned (core + anomalies + arc milestones). Denominator of the ratio.
    pub vectors_returned: usize,
    /// Anomalies (pattern breaks) included, trimmed to budget.
    pub anomalies_included: usize,
    /// Evolutionary arc milestones included (t, drift): the subject's trajectory over time.
    pub arc_points: Vec<(f64, f32)>,
    /// Lexical label of the **current** dominant behaviour (what interests the subject now).
    pub core_label: String,
    /// Lexical labels aligned with `arc_points`: what occupied the subject at each milestone.
    pub arc_labels: Vec<String>,
    /// Lexical labels of the included anomalies.
    pub anomaly_labels: Vec<String>,
    /// Trajectories **per behaviour**: for the most prevalent domains, their fraction of
    /// activity at each arc milestone. Allows narrating "X rose, fell and returned" for a
    /// concrete behaviour, not just the global centroid. Closes the domain-reversal gap.
    pub domain_arcs: Vec<(String, Vec<f32>)>,
    /// Histogram `(label, count)` for each returned milestone, ALIGNED with `arc_points`. Additive
    /// (Improvement E): allows the prose layer to derive a label via **common terms** (TF-IDF)
    /// instead of a single representative text, without the core deciding presentation. Does not
    /// alter any other engine signal.
    pub arc_label_histograms: Vec<Vec<(String, usize)>>,
    /// Token estimate of the block, guaranteed ≤ token_budget.
    pub token_estimate: usize,
    /// Label of the mode focused on by `EVOKE … RESONATING WITH { trait }`
    /// (the aspect of the subject that resonates with the trait). `None` if no RESONATING WITH clause.
    pub resonating_mode: Option<String>,
    /// **Per-mode trajectory**: `(label, drift)` for each live mode — how far that
    /// behaviour has shifted since it was born. Complements `arc_points` (global centroid drift) with
    /// the evolution of *each mode*, not the blind average. See [`crate::Mode::drift`].
    pub mode_drifts: Vec<(String, f32)>,
}

impl CompressedContext {
    /// Compression ratio: perceptions represented / vectors returned.
    pub fn compression_ratio(&self) -> f64 {
        if self.vectors_returned == 0 {
            return 0.0;
        }
        self.represented as f64 / self.vectors_returned as f64
    }
}

/// Default cost, in tokens, of a dense vector when rendered for the LLM. This is a **declared
/// default**, not a magic constant buried in the algorithm: [`EvokeRequest::tokens_per_vector`]
/// overrides it with the real mean measured by tiktoken in the orchestration layer. (Resolves
/// debt #1 of TRUTH 100%: the cost is measured or injected, never invented.)
pub const DEFAULT_TOKENS_PER_VECTOR: usize = 24;

/// `EVOKE`: resolves the essence of a subject within the token budget.
///
/// Returns `None` if the subject has no live archetype.
pub fn evoke(store: &ArchetypeStore, req: &EvokeRequest, now: Tick) -> Option<CompressedContext> {
    let a: &Archetype = store.get(&req.subject)?;
    if a.trace.weight(now) < DEFAULT_THETA_FADE {
        return None; // essence has faded
    }

    // Budget allocation: core (1) + arc (trajectory is the subject's signature over time,
    // priority over anomalies) + loose anomalies with whatever remains.
    let budget_vectors = req.token_budget / req.tokens_per_vector.max(1);
    let core_vectors = 1usize;
    let base_arc_quota = (budget_vectors.saturating_sub(core_vectors)) * 2 / 3;
    // The requested detail (RESOLUTION/PROJECTING) modulates how much arc we return.
    let arc_quota = match req.arc_detail {
        ArcDetail::None => 0,
        ArcDetail::Summary => base_arc_quota.min(4),
        ArcDetail::Full => base_arc_quota,
    };
    let (arc_points, arc_labels, arc_label_histograms) = arc_signature(a, arc_quota, req.since);
    let arc_count = arc_points.len();

    let room_for_anomalies = budget_vectors
        .saturating_sub(core_vectors)
        .saturating_sub(arc_count);
    let anomalies_included = a.anomalies.len().min(room_for_anomalies);
    // Labels of included anomalies (aligned; robust when fewer labels than vectors).
    let anomaly_labels: Vec<String> = a
        .anomaly_labels
        .iter()
        .take(anomalies_included)
        .cloned()
        .collect();

    let vectors_returned = core_vectors + arc_count + anomalies_included;
    let token_estimate = vectors_returned * req.tokens_per_vector;

    // Domain trajectories: most relevant by **peak** (not accumulated popularity, see
    // `domain_trajectories`). The cap is **derived from the budget** (`budget_vectors`), not a
    // magic constant: more budget → more domains fit in the prose. (TRUTH 100% debt #4 resolved.)
    let domain_arcs = if req.arc_detail == ArcDetail::None {
        Vec::new()
    } else {
        domain_trajectories(a, budget_vectors.max(1), req.since)
    };

    Some(CompressedContext {
        subject: a.subject.clone(),
        represented: a.represented,
        vectors_returned,
        anomalies_included,
        arc_points,
        core_label: a.core_label.clone(),
        arc_labels,
        anomaly_labels,
        domain_arcs,
        arc_label_histograms,
        token_estimate,
        resonating_mode: None, // set by the executor if the statement has RESONATING WITH.
        // Per-mode trajectory: drift of each live mode (how far that behaviour shifted from its origin).
        mode_drifts: a
            .modes
            .iter()
            .filter(|m| m.trace.weight(now) >= DEFAULT_THETA_FADE)
            .map(|m| (m.label.clone(), m.drift()))
            .collect(),
    })
}

/// Reconstructs, for the `max_domains` most relevant behaviours, their **fraction of activity
/// at each milestone** of the arc (time series normalised per milestone). This is the source of
/// "did X rise/fall/return?".
///
/// **Ranking (Improvement B, 2026-06-12).** Previously we used `total_count` (sum of appearances).
/// That flattened peaks: a domain appearing uniformly across all phases always won, while one with
/// a clear PEAK in a single phase fell outside the top-K. For questions like "was X ever important
/// even if not now?" that destroyed signal in item-centric verticals (unique titles that don't
/// repeat across phases).
///
/// New ranking: `score = max_phase × (1 + variance_across_phases)`. Elevates domains that had a
/// clear peak (`max_phase` high) AND were concentrated in few phases (`variance` high).
/// Penalises uniform domains and rescues the vanished ones.
fn domain_trajectories(
    a: &Archetype,
    max_domains: usize,
    since: Option<Tick>,
) -> Vec<(String, Vec<f32>)> {
    use std::collections::HashSet;
    let milestones: Vec<&crate::archetype::ArcMilestone> = a
        .arc
        .iter()
        .filter(|m| since.is_none_or(|s| m.at >= s))
        .collect();
    if milestones.is_empty() {
        return Vec::new();
    }
    // Universe of labels that appeared at least once in any milestone.
    let mut universe: HashSet<&str> = HashSet::new();
    for m in &milestones {
        for (label, _) in &m.label_histogram {
            universe.insert(label.as_str());
        }
    }
    // Pre-compute per-milestone totals (denominators) once.
    let phase_totals: Vec<usize> = milestones
        .iter()
        .map(|m| m.label_histogram.iter().map(|(_, c)| *c).sum())
        .collect();

    // For each domain, compute the normalised series AND the peak+variance score.
    let mut scored: Vec<(&str, Vec<f32>, f32)> = universe
        .into_iter()
        .map(|label| {
            let series: Vec<f32> = milestones
                .iter()
                .zip(phase_totals.iter())
                .map(|(m, &total)| {
                    let c = m
                        .label_histogram
                        .iter()
                        .find(|(l, _)| l.as_str() == label)
                        .map(|(_, c)| *c)
                        .unwrap_or(0);
                    if total > 0 {
                        c as f32 / total as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            let max_phase = series.iter().cloned().fold(0.0_f32, f32::max);
            // Simple population variance. With n=4 phases it is stable and cheap.
            let n = series.len().max(1) as f32;
            let mean = series.iter().sum::<f32>() / n;
            let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let score = max_phase * (1.0 + variance);
            (label, series, score)
        })
        .collect();

    // Stable order: score desc, alphabetical tiebreak for reproducibility.
    scored.sort_by(|(la, _, sa), (lb, _, sb)| {
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| la.cmp(lb))
    });
    scored.truncate(max_domains);

    scored
        .into_iter()
        .map(|(label, series, _)| (label.to_string(), series))
        .collect()
}

/// Reduces the arc to `quota` milestones (by uniform *thinning*) and projects them as
/// `(t, accumulated_drift)` — drift = `1 - cos(milestone_i, milestone_0)` ∈ [0, 2]. Milestone_0
/// (absolute origin of identity) serves as reference, even if `since` filters which milestones are reported.
type ArcSignature = (Vec<(f64, f32)>, Vec<String>, Vec<Vec<(String, usize)>>);

fn arc_signature(a: &Archetype, quota: usize, since: Option<Tick>) -> ArcSignature {
    if quota == 0 || a.arc.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let origin = &a.arc[0].direction;
    // `ACROSS span`: only the milestones within the requested time window.
    let window: Vec<&crate::archetype::ArcMilestone> = a
        .arc
        .iter()
        .filter(|m| since.is_none_or(|s| m.at >= s))
        .collect();
    if window.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let n = window.len();
    let mut points = Vec::with_capacity(quota.min(n));
    let mut labels = Vec::with_capacity(quota.min(n));
    let mut histograms = Vec::with_capacity(quota.min(n));
    let mut push = |m: &crate::archetype::ArcMilestone| {
        points.push((m.at, 1.0 - cosine(&m.direction, origin)));
        labels.push(m.label.clone());
        histograms.push(m.label_histogram.clone());
    };
    if n <= quota {
        for m in &window {
            push(m);
        }
    } else {
        // Uniform thinning: `quota` evenly-spaced milestones (includes first and last of window).
        for i in 0..quota {
            let idx = (i * (n - 1)) / (quota - 1).max(1);
            push(window[idx]);
        }
    }
    (points, labels, histograms)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified EVOKE (L6): the bi-layer in a single evocation, under a single budget.
// ─────────────────────────────────────────────────────────────────────────────

/// Token estimate of a text by counting words (whitespace-separated units). This is a **real
/// measurement of the text**, declared as an approximation: the exact tokenizer count of the LLM
/// is injected via the `fact_cost` parameter of [`evoke_unified`]. Not an invented constant —
/// either the text is measured, or tiktoken is injected.
pub fn approx_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// **Unified** context: the characterological signature (layer-2, gist) and the exact episodic
/// facts (layer-1) that ONE evocation gathers under ONE budget. Answers character AND nominal
/// without hand-stitching two systems in the orchestration layer — it is the bi-layer of
/// *Complementary Learning Systems* exposed by the engine in a single query.
#[derive(Debug, Clone)]
pub struct UnifiedContext {
    /// The characterological essence (layer-2). `None` if no live archetype or the remaining
    /// budget does not cover even the core (facts consumed it).
    pub gist: Option<CompressedContext>,
    /// Exact facts (layer-1) included, ordered by physical score (`relevance · life`).
    pub facts: Vec<RecalledFact>,
    /// Tokens consumed by facts (sum of the real `fact_cost` of each included text).
    pub fact_tokens: usize,
    /// Total tokens of the block (gist + facts). Guaranteed ≤ `req.token_budget`.
    pub total_tokens: usize,
}

/// Unified `EVOKE`: distributes ONE budget between layer-1 (exact facts) and layer-2 (gist).
///
/// `fact_budget` (clamped to `req.token_budget`) is the portion reserved for facts; the rest goes
/// to the gist. Facts are chosen **greedy by physical score** (see [`FactStore::search`]) filling
/// up to `fact_budget` according to `fact_cost` — the real tokenizer injected by the caller. The
/// gist is evoked with the remaining budget only if it covers at least the core. It is **read-only**
/// (composable, no side effects): for reinforcement on evocation (spaced repetition) use
/// [`FactStore::recall`]. Guarantees `total_tokens ≤ req.token_budget`.
#[allow(clippy::too_many_arguments)]
pub fn evoke_unified(
    archetypes: &ArchetypeStore,
    facts: &FactStore,
    req: &EvokeRequest,
    query: &[f32],
    fact_budget: usize,
    now: Tick,
    fact_cost: impl Fn(&str) -> usize,
) -> UnifiedContext {
    let fact_budget = fact_budget.min(req.token_budget);

    // Layer-1: facts by physical score, greedy knapsack under `fact_budget` with measured real cost.
    // (Greedy by score filling gaps: an honest approximation, not claimed optimal.)
    let ranked = facts.search(&req.subject, query, facts.len(), now, DEFAULT_THETA_FADE);
    let mut chosen = Vec::new();
    let mut fact_tokens = 0usize;
    for (score, f) in ranked {
        let cost = fact_cost(&f.text);
        if fact_tokens + cost <= fact_budget {
            fact_tokens += cost;
            chosen.push(RecalledFact {
                text: f.text.clone(),
                provenance: f.provenance.clone(),
                score,
            });
        }
    }

    // Layer-2: the gist with what remains, only if it covers at least the core (otherwise facts consumed it).
    let gist_budget = req.token_budget - fact_tokens;
    let gist = if gist_budget >= req.tokens_per_vector {
        let gist_req = EvokeRequest {
            token_budget: gist_budget,
            ..req.clone()
        };
        evoke(archetypes, &gist_req, now)
    } else {
        None
    };

    let total_tokens = fact_tokens + gist.as_ref().map_or(0, |g| g.token_estimate);
    UnifiedContext {
        gist,
        facts: chosen,
        fact_tokens,
        total_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archetype::{ArchetypeStore, Resilience};
    use crate::factstore::FactStore;
    use crate::synthesis::IntentionVector;

    fn store_with(absorbed: usize, anomalies: usize) -> ArchetypeStore {
        let mut s = ArchetypeStore::new();
        s.imprint(
            &IntentionVector {
                subject: "user:Xolotl".into(),
                centroid: vec![1.0, 0.0],
                anomalies: vec![vec![0.0, 1.0]; anomalies],
                core_label: "core".into(),
                anomaly_labels: vec!["novelty".into(); anomalies],
                absorbed,
                redundant: 0,
                label_histogram: vec![("core".into(), absorbed)],
                modes: vec![],
            },
            Resilience::High,
            0.0,
        );
        s
    }

    #[test]
    fn evoke_unknown_subject_is_none() {
        let s = ArchetypeStore::new();
        let req = EvokeRequest::new("ghost", 800);
        assert!(evoke(&s, &req, 0.0).is_none());
    }

    #[test]
    fn unified_evoke_answers_character_and_nominal_under_one_budget() {
        // Layer-2: a subject with consolidated essence (character gist).
        let archetypes = store_with(10_000, 2);
        // Layer-1: an exact fact — the nominal info a gist (an average) could never store.
        let mut facts = FactStore::new();
        facts.remember(
            "user:Xolotl",
            "API key is sk-ABC123",
            vec![0.0, 1.0],
            "agent",
            1.0,
            86_400.0,
            0.0,
        );

        let req = EvokeRequest::new("user:Xolotl", 800);
        let u = evoke_unified(
            &archetypes,
            &facts,
            &req,
            &[0.0, 1.0],
            100,
            0.0,
            approx_token_count,
        );

        // A single evocation answers BOTH question types:
        assert!(u.gist.is_some(), "character: layer-2 is present");
        assert_eq!(u.facts.len(), 1, "nominal: the exact fact was included");
        assert_eq!(
            u.facts[0].text, "API key is sk-ABC123",
            "layer-1 lossless, verbatim"
        );
        // …under ONE budget, respected.
        assert!(u.total_tokens <= 800, "total {} ≤ 800", u.total_tokens);
        assert_eq!(
            u.total_tokens,
            u.fact_tokens + u.gist.as_ref().unwrap().token_estimate
        );
    }

    #[test]
    fn unified_evoke_fact_cost_is_injected_not_hardcoded() {
        // No archetype: isolates layer-1 logic (gist will be None, does not affect facts).
        let archetypes = ArchetypeStore::new();
        let mut facts = FactStore::new();
        facts.remember("u", "one two", vec![1.0, 0.0], "a", 1.0, 86_400.0, 0.0);
        facts.remember("u", "three four", vec![0.7, 0.7], "a", 1.0, 86_400.0, 0.0);
        facts.remember("u", "five six", vec![0.0, 1.0], "a", 1.0, 86_400.0, 0.0);
        let req = EvokeRequest::new("u", 800);
        let q = [0.7, 0.7];

        // Cheap cost (1 token/fact) → all 3 fit in fact_budget=3.
        let cheap = evoke_unified(&archetypes, &facts, &req, &q, 3, 0.0, |_| 1);
        assert_eq!(cheap.facts.len(), 3);
        assert_eq!(cheap.fact_tokens, 3);

        // Same budget, expensive cost (5 > 3) → none fit. The cost rules, not a baked-in 24.
        let pricey = evoke_unified(&archetypes, &facts, &req, &q, 3, 0.0, |_| 5);
        assert_eq!(pricey.facts.len(), 0);

        // Budget 5, cost 5 → exactly one fits (the one with the highest physical score).
        let one = evoke_unified(&archetypes, &facts, &req, &q, 5, 0.0, |_| 5);
        assert_eq!(one.facts.len(), 1);
    }

    #[test]
    fn unified_skips_gist_when_facts_consume_budget() {
        let archetypes = store_with(10_000, 0); // gist available for "user:Xolotl"
        let mut facts = FactStore::new();
        facts.remember("user:Xolotl", "x", vec![1.0, 0.0], "a", 1.0, 86_400.0, 0.0);
        let req = EvokeRequest::new("user:Xolotl", 30); // small budget

        // The fact costs 25 → 5 remain < tokens_per_vector(24) ⇒ gist is skipped, budget intact.
        let u = evoke_unified(&archetypes, &facts, &req, &[1.0, 0.0], 30, 0.0, |_| 25);
        assert_eq!(u.facts.len(), 1);
        assert!(
            u.gist.is_none(),
            "facts consumed the budget: no gist"
        );
        assert_eq!(u.total_tokens, 25);
        assert!(u.total_tokens <= 30);
    }

    #[test]
    fn evoke_respects_token_budget() {
        let s = store_with(100_000, 1000);
        let req = EvokeRequest::new("user:Xolotl", 800);
        let ctx = evoke(&s, &req, 0.0).unwrap();
        assert!(
            ctx.token_estimate <= 800,
            "context fits within the budget"
        );
    }

    #[test]
    fn evoke_achieves_massive_compression() {
        let s = store_with(100_000, 5);
        let req = EvokeRequest::new("user:Xolotl", 800);
        let ctx = evoke(&s, &req, 0.0).unwrap();
        assert_eq!(ctx.represented, 100_000);
        assert!(ctx.compression_ratio() > 1000.0, "compression >> 1000:1");
    }

    #[test]
    fn evoke_reports_per_mode_drift() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};
        let mut s = ArchetypeStore::new();
        // Cycle 1 at [1,0] → the "x" mode is born (origin [1,0]).
        let ps1: Vec<Perception> = (0..6)
            .map(|_| Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "x"))
            .collect();
        let r1: Vec<&Perception> = ps1.iter().collect();
        s.imprint(
            &distill("u", &r1, DistillConfig::default()).unwrap(),
            Resilience::High,
            0.0,
        );
        // Cycle 2 shifted to [0.6,0.8] (merges into the same mode) → the mode drifts.
        let ps2: Vec<Perception> = (0..6)
            .map(|_| Perception::new("u", vec![0.6, 0.8], 1.0, 3600.0, 0.0).with_trait("act", "x"))
            .collect();
        let r2: Vec<&Perception> = ps2.iter().collect();
        s.imprint(
            &distill("u", &r2, DistillConfig::default()).unwrap(),
            Resilience::High,
            1.0,
        );

        let ctx = evoke(&s, &EvokeRequest::new("u", 800), 1.0).unwrap();
        assert_eq!(
            ctx.mode_drifts.len(),
            1,
            "one live mode → one per-mode trajectory"
        );
        assert_eq!(ctx.mode_drifts[0].0, "x");
        assert!(
            ctx.mode_drifts[0].1 > 0.0,
            "mode drifted from its origin: {:?}",
            ctx.mode_drifts
        );
    }

    /// Store with an arc of several milestones at increasing times (to test span/resolution).
    fn store_with_arc() -> ArchetypeStore {
        let mut s = ArchetypeStore::new();
        // 4 sleep cycles at t = 0, 100, 200, 300 with drifting directions.
        let dirs = [
            vec![1.0, 0.0],
            vec![0.8, 0.6],
            vec![0.0, 1.0],
            vec![-0.6, 0.8],
        ];
        for (i, d) in dirs.iter().enumerate() {
            s.imprint(
                &IntentionVector {
                    subject: "u".into(),
                    centroid: d.clone(),
                    anomalies: vec![],
                    core_label: "dom".into(),
                    anomaly_labels: vec![],
                    absorbed: 10,
                    redundant: 0,
                    label_histogram: vec![("dom".into(), 10)],
                    modes: vec![],
                },
                Resilience::High,
                i as f64 * 100.0,
            );
        }
        s
    }

    #[test]
    fn resolution_point_returns_no_arc() {
        let s = store_with_arc();
        let req = EvokeRequest {
            arc_detail: ArcDetail::None,
            ..EvokeRequest::new("u", 800)
        };
        let ctx = evoke(&s, &req, 300.0).unwrap();
        assert!(
            ctx.arc_points.is_empty(),
            "RESOLUTION point / snapshot ⇒ no arc"
        );
    }

    #[test]
    fn resolution_summary_caps_arc_points() {
        let s = store_with_arc();
        let full = evoke(&s, &EvokeRequest::new("u", 800), 300.0).unwrap();
        let summary = evoke(
            &s,
            &EvokeRequest {
                arc_detail: ArcDetail::Summary,
                ..EvokeRequest::new("u", 800)
            },
            300.0,
        )
        .unwrap();
        assert!(summary.arc_points.len() <= 4);
        assert!(summary.arc_points.len() <= full.arc_points.len());
    }

    #[test]
    fn span_window_filters_old_milestones() {
        let s = store_with_arc();
        // since = 150 ⇒ only the milestones at t=200 and t=300 are included.
        let req = EvokeRequest {
            since: Some(150.0),
            ..EvokeRequest::new("u", 800)
        };
        let ctx = evoke(&s, &req, 300.0).unwrap();
        assert!(
            ctx.arc_points.iter().all(|(t, _)| *t >= 150.0),
            "{:?}",
            ctx.arc_points
        );
        assert_eq!(ctx.arc_points.len(), 2);
    }

    #[test]
    fn domain_arcs_capture_per_domain_reversal() {
        // The gap the adversarial benchmark exposed: the GLOBAL arc doesn't distinguish which concrete
        // behaviour returned. `domain_arcs` reconstructs per-domain prevalence at each milestone.
        // Script: "yoga" high → falls → falls → RETURNS; "trail" grows monotonically.
        let cycles = [
            vec![("yoga".to_string(), 7usize), ("trail".to_string(), 3)],
            vec![("yoga".to_string(), 2), ("trail".to_string(), 8)],
            vec![("yoga".to_string(), 1), ("trail".to_string(), 9)],
            vec![("yoga".to_string(), 6), ("trail".to_string(), 4)],
        ];
        let mut s = ArchetypeStore::new();
        for (i, hist) in cycles.iter().enumerate() {
            let absorbed: usize = hist.iter().map(|(_, c)| c).sum();
            s.imprint(
                &IntentionVector {
                    subject: "u".into(),
                    centroid: vec![1.0, 0.0],
                    anomalies: vec![],
                    core_label: hist[0].0.clone(),
                    anomaly_labels: vec![],
                    absorbed,
                    redundant: 0,
                    label_histogram: hist.clone(),
                    modes: vec![],
                },
                Resilience::High,
                i as f64 * 100.0,
            );
        }
        let ctx = evoke(&s, &EvokeRequest::new("u", 800), 400.0).unwrap();

        let yoga = &ctx
            .domain_arcs
            .iter()
            .find(|(l, _)| l == "yoga")
            .expect("yoga series")
            .1;
        // Yoga prevalence per milestone: high, low, low, high ⇒ round trip (what the global arc missed).
        assert!(
            yoga[0] > 0.5 && yoga[1] < 0.3 && yoga[3] > 0.4,
            "yoga reversal: {yoga:?}"
        );

        let trail = &ctx
            .domain_arcs
            .iter()
            .find(|(l, _)| l == "trail")
            .expect("trail series")
            .1;
        assert!(
            trail[0] < 0.4 && trail[2] > 0.8,
            "trail rising: {trail:?}"
        );
    }
}
