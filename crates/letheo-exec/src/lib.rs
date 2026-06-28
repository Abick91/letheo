//! # letheo-exec · MQL executor
//!
//! Closes the language loop: takes the AST produced by `letheo-mql` and translates it into real
//! operations on `CognitiveRuntime`. The orchestration layer (Python, CLI, agent) no longer needs
//! to know the Rust core API — it speaks MQL.
//!
//! Biological mapping:
//! - `PERCEIVE` → `Runtime::perceive(...)` with Provider embedding (traits as text).
//! - `DISTILL`  → `Runtime::breathe([subject])` (one dream cycle for that subject).
//! - `EVOKE`    → `Runtime::evoke(...)` with the statement's token budget.
//! - `FADE`     → semantic GC sweep (performed inside the next `breathe`).
//! - `IMPRINT`  → consolidates/anchors the subject's archetype (reinforces its physics and modes).

use letheo_core::{
    ArcDetail, BreathReport, CognitiveRuntime, CompressedContext, EvokeRequest, Fact, Perception,
    RecalledFact, Tick,
};
use letheo_inference::Provider;
use letheo_mql::ast::{
    Distill, Evoke, Facts, Fade, Field, Imprint, Perceive, Predicate, Projection, Recall,
    Reinforce, Resilience, Resolution, Statement,
};

/// Result of executing one MQL statement.
// `Evoked` carries a large `CompressedContext` compared to the other variants; it is an ephemeral
// result type (one per statement), so the size difference is harmless — not worth boxing.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ExecResult {
    Perceived {
        subject: String,
    },
    Dreamed(BreathReport),
    Evoked(CompressedContext),
    Faded {
        swept: usize,
    },
    Imprinted {
        archetype: String,
        note: &'static str,
    },
    /// `RECALL`: episodic facts retrieved (verbatim), ranked by physics. Read-only.
    Recalled(Vec<RecalledFact>),
    /// `REINFORCE`: how many facts were reinforced (their decay was reset).
    Reinforced {
        count: usize,
    },
}

/// Execution errors (separate from parse errors).
#[derive(Debug, Clone)]
pub enum ExecError {
    NoSuchSubject(String),
    MissingBudget,
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecError::NoSuchSubject(s) => write!(f, "no live archetype for '{s}'"),
            ExecError::MissingBudget => write!(f, "EVOKE requires WITHIN budget N tokens"),
        }
    }
}

impl std::error::Error for ExecError {}

/// The executor holds a runtime and an embedding provider. Not `Sync` due to the provider.
pub struct Executor<P: Provider> {
    rt: CognitiveRuntime,
    provider: P,
}

impl<P: Provider> Executor<P> {
    pub fn new(rt: CognitiveRuntime, provider: P) -> Self {
        Self { rt, provider }
    }

    pub fn runtime(&self) -> &CognitiveRuntime {
        &self.rt
    }

    pub fn runtime_mut(&mut self) -> &mut CognitiveRuntime {
        &mut self.rt
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Executes one MQL statement against the runtime at logical tick `now`.
    pub fn execute(&mut self, stmt: &Statement, now: Tick) -> Result<ExecResult, ExecError> {
        match stmt {
            Statement::Perceive(p) => self.exec_perceive(p, now),
            Statement::Distill(d) => self.exec_distill(d, now),
            Statement::Evoke(e) => self.exec_evoke(e, now),
            Statement::Fade(f) => self.exec_fade(f, now),
            Statement::Imprint(i) => self.exec_imprint(i, now),
            Statement::Recall(r) => self.exec_recall(r, now),
            Statement::Reinforce(r) => self.exec_reinforce(r, now),
        }
    }

    /// Executes a full program (multiple statements) and returns all results.
    pub fn execute_program(
        &mut self,
        stmts: &[Statement],
        now: Tick,
    ) -> Vec<Result<ExecResult, ExecError>> {
        stmts.iter().map(|s| self.execute(s, now)).collect()
    }

    fn exec_perceive(&mut self, p: &Perceive, now: Tick) -> Result<ExecResult, ExecError> {
        // Raw stimulus: traits concatenated as text and embedded by the provider.
        // Traits *are* the stimulus (no binary payload in the AST).
        let text = traits_to_text(p);
        let embedding = self.provider.embed(&text);
        let salience = p.salience.unwrap_or(0.5);
        let halflife = p.halflife.map(|d| d.seconds).unwrap_or(86_400.0); // default 1 day
        let mut perception = Perception::new(&p.subject, embedding, salience, halflife, now);
        for (k, v) in &p.traits {
            perception = perception.with_trait(k.clone(), v.clone());
        }
        self.rt.perceive(perception);
        Ok(ExecResult::Perceived {
            subject: p.subject.clone(),
        })
    }

    fn exec_distill(&mut self, d: &Distill, now: Tick) -> Result<ExecResult, ExecError> {
        let report = match &d.filter {
            None => self.rt.breathe(&[&d.subject], now),
            Some(pred) => self
                .rt
                .breathe_where(&[&d.subject], now, |p| eval_on(pred, p, now)),
        };
        Ok(ExecResult::Dreamed(report))
    }

    fn exec_evoke(&mut self, e: &Evoke, now: Tick) -> Result<ExecResult, ExecError> {
        let token_budget = e.token_budget.ok_or(ExecError::MissingBudget)?;

        // `ACROSS span D` → time window: only arc milestones with at ≥ now − D.
        let since = e.span.map(|d| (now - d.seconds).max(0.0));

        // `RESOLUTION` overrides `PROJECTING`; if neither is set, full arc.
        let arc_detail = match (e.resolution, e.projecting) {
            (Some(Resolution::Point), _) => ArcDetail::None,
            (Some(Resolution::Summary), _) => ArcDetail::Summary,
            (Some(Resolution::Arc), _) => ArcDetail::Full,
            (None, Some(Projection::Snapshot)) => ArcDetail::None,
            (None, Some(Projection::Trajectory)) => ArcDetail::Full,
            (None, None) => ArcDetail::Full,
        };

        let req = EvokeRequest {
            subject: e.subject.clone(),
            token_budget,
            since,
            arc_detail,
            ..EvokeRequest::new(e.subject.clone(), token_budget)
        };
        let mut ctx = self
            .rt
            .evoke(&req, now)
            .ok_or_else(|| ExecError::NoSuchSubject(e.subject.clone()))?;

        // `RESONATING WITH { traits }`: no longer ignored. We embed the traits with the real provider
        // and focus evocation on the **mode** of the subject that resonates with them (the relevant
        // aspect, not the global dominant behaviour).
        if !e.resonating_with.is_empty() {
            let query = self.provider.embed(&e.resonating_with.join(" "));
            ctx.resonating_mode = self
                .rt
                .long_term()
                .get(&e.subject)
                .and_then(|a| a.resonant_mode_label(&query));
        }
        Ok(ExecResult::Evoked(ctx))
    }

    fn exec_fade(&mut self, f: &Fade, now: Tick) -> Result<ExecResult, ExecError> {
        let swept = match &f.filter {
            // No WHERE: sweep at the default forgetting threshold (physics decides what falls).
            None => self
                .rt
                .fade_only(now, letheo_core::entropy::DEFAULT_THETA_FADE),
            // With WHERE: the user predicate *is* the forgetting condition.
            Some(pred) => self.rt.fade_where(|p| eval_on(pred, p, now)),
        };
        Ok(ExecResult::Faded { swept })
    }

    fn exec_imprint(&mut self, i: &Imprint, now: Tick) -> Result<ExecResult, ExecError> {
        // Real IMPRINT: **consolidates/anchors** the named subject's archetype — reinforces its
        // physics (and its modes') to gain permanence. RESILIENCE maps to how much is consolidated
        // (higher resilience → larger λ reduction).
        let consolidation = match i.resilience {
            Some(Resilience::High) => 0.2,
            Some(Resilience::Medium) => 0.1,
            Some(Resilience::Low) | None => 0.05,
        };
        if self.rt.consolidate(&i.archetype, now, consolidation) {
            Ok(ExecResult::Imprinted {
                archetype: i.archetype.clone(),
                note: "essence consolidated: Δt→0 and λ reduced (permanence gained)",
            })
        } else {
            Err(ExecError::NoSuchSubject(i.archetype.clone()))
        }
    }

    /// `RECALL` (layer-1): retrieves the subject's episodic facts that resonate with the query,
    /// ranked by physics (`relevance · life`) and truncated to top-k. **Read-only** (does not
    /// reinforce — `REINFORCE` does that). Optional `WHERE` filters by `resonates`/`weight`/`age`/`salience`.
    fn exec_recall(&mut self, r: &Recall, now: Tick) -> Result<ExecResult, ExecError> {
        let query = self.provider.embed(&r.resonating_with.join(" "));
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        let mut scored: Vec<(f64, RecalledFact)> = self
            .rt
            .facts()
            .alive_for(&r.subject, now, theta)
            .filter(|f| match &r.filter {
                None => true,
                Some(pred) => pred.eval(&FactFacts {
                    f,
                    query: &query,
                    now,
                }),
            })
            .map(|f| {
                let relevance = letheo_core::vector::cosine(&f.embedding, &query).max(0.0) as f64;
                let score = relevance * f.trace.weight(now);
                (
                    score,
                    RecalledFact {
                        text: f.text.clone(),
                        provenance: f.provenance.clone(),
                        score,
                    },
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(r.k);
        Ok(ExecResult::Recalled(
            scored.into_iter().map(|(_, rf)| rf).collect(),
        ))
    }

    /// `REINFORCE` (layer-1): retrieves the top-k facts resonating with the query and **reinforces
    /// them** (resets their decay → spaced repetition). Mutates layer-1. Returns how many were reinforced.
    fn exec_reinforce(&mut self, r: &Reinforce, now: Tick) -> Result<ExecResult, ExecError> {
        let query = self.provider.embed(&r.resonating_with.join(" "));
        let reinforced = self.rt.recall(&r.subject, &query, r.k, now);
        Ok(ExecResult::Reinforced {
            count: reinforced.len(),
        })
    }
}

/// Bridge between an MQL predicate and a concrete runtime perception. Keeps `letheo-mql` ignorant
/// of `letheo-core`: `WHERE` semantics live in the AST, physical data lives here.
struct PerceptionFacts<'a> {
    p: &'a Perception,
    now: Tick,
}

impl Facts for PerceptionFacts<'_> {
    fn numeric(&self, field: &Field) -> Option<f64> {
        match field {
            Field::Weight => Some(self.p.weight(self.now)),
            Field::Salience => Some(self.p.trace.salience),
            Field::Age => Some(self.p.trace.delta_t(self.now)),
            // No query in the perception context (DISTILL/FADE): resonance is not available.
            Field::Resonance => None,
            Field::Trait(_) => None,
        }
    }

    fn text(&self, key: &str) -> Option<String> {
        self.p.traits.get(key).cloned()
    }
}

/// Evaluates a `WHERE` predicate against a perception at tick `now`.
fn eval_on(pred: &Predicate, p: &Perception, now: Tick) -> bool {
    pred.eval(&PerceptionFacts { p, now })
}

/// Bridge between an MQL predicate and an **episodic fact** (layer-1) + the statement query. Gives
/// meaning to `WHERE resonates > θ`: resonance is the cosine of the fact against the embedded query.
/// Facts have no trait map → they expose only physics (`weight`/`salience`/`age`) and `resonance`.
struct FactFacts<'a> {
    f: &'a Fact,
    query: &'a [f32],
    now: Tick,
}

impl Facts for FactFacts<'_> {
    fn numeric(&self, field: &Field) -> Option<f64> {
        match field {
            Field::Weight => Some(self.f.trace.weight(self.now)),
            Field::Salience => Some(self.f.trace.salience),
            Field::Age => Some(self.f.trace.delta_t(self.now)),
            Field::Resonance => {
                Some(letheo_core::vector::cosine(&self.f.embedding, self.query) as f64)
            }
            Field::Trait(_) => None,
        }
    }

    fn text(&self, _key: &str) -> Option<String> {
        None
    }
}

/// Concatenates traits from a PERCEIVE statement into a single text line for embedding.
/// Stable: iterates in alphabetical key order (BTreeMap already does this).
fn traits_to_text(p: &Perceive) -> String {
    let parts: Vec<String> = p.traits.iter().map(|(k, v)| format!("{k} {v}")).collect();
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_core::RuntimeConfig;
    use letheo_inference::MockProvider;
    use letheo_mql::parse;

    fn fresh() -> Executor<MockProvider> {
        Executor::new(
            CognitiveRuntime::new(RuntimeConfig::default()),
            MockProvider::new(),
        )
    }

    /// Registers a fact (layer-1) with an embedding of `query_text` (so a query with those tokens
    /// retrieves it) and the given `halflife`. Layer-1 is written via API; MQL queries it.
    fn remember_fact(
        ex: &mut Executor<MockProvider>,
        subject: &str,
        text: &str,
        query_text: &str,
        halflife: f64,
        now: f64,
    ) {
        let emb = ex.provider().embed(query_text);
        ex.runtime_mut()
            .remember(subject, text, emb, "test", 1.0, halflife, now);
    }

    #[test]
    fn full_program_perceive_distill_evoke() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { act: purchase, object: shoes }
            PERCEIVE interaction FROM subject "u:X" AS { act: purchase, object: shoes }
            PERCEIVE interaction FROM subject "u:X" AS { act: purchase, object: shoes }
            DISTILL subject "u:X" INTO intention_vector COMPRESSING BY semantic_variance
            EVOKE essence OF "u:X" WITHIN budget 800 tokens
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        assert_eq!(results.len(), 5);

        // 3 accepted perceptions
        for r in &results[..3] {
            assert!(matches!(r, Ok(ExecResult::Perceived { .. })));
        }
        // DISTILL → BreathReport with 1 consolidated subject
        match &results[3] {
            Ok(ExecResult::Dreamed(r)) => {
                assert_eq!(r.distilled_subjects, 1);
                assert_eq!(r.perceptions_absorbed, 3);
            }
            other => panic!("expected Dreamed, got {other:?}"),
        }
        // EVOKE → context representing those 3 events
        match &results[4] {
            Ok(ExecResult::Evoked(ctx)) => {
                assert_eq!(ctx.represented, 3);
                assert!(ctx.token_estimate <= 800);
            }
            other => panic!("expected Evoked, got {other:?}"),
        }
    }

    #[test]
    fn evoke_missing_budget_fails() {
        let stmts = parse(r#"EVOKE essence OF "u:X""#).unwrap();
        let mut ex = fresh();
        let r = &ex.execute_program(&stmts, 0.0)[0];
        assert!(matches!(r, Err(ExecError::MissingBudget)));
    }

    #[test]
    fn evoke_unknown_subject_fails() {
        let stmts = parse(r#"EVOKE essence OF "ghost" WITHIN budget 800 tokens"#).unwrap();
        let mut ex = fresh();
        let r = &ex.execute_program(&stmts, 0.0)[0];
        assert!(matches!(r, Err(ExecError::NoSuchSubject(_))));
    }

    #[test]
    fn fade_sweeps_decayed_noise() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { a: b } WITH salience 0.1 DECAYS halflife 1h
            FADE noise PRESERVING archetype_contribution
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        // PERCEIVE at t=0 and FADE 10h later → the weak event must fall.
        ex.execute(&stmts[0], 0.0).unwrap();
        let r = ex.execute(&stmts[1], 3600.0 * 10.0).unwrap();
        match r {
            ExecResult::Faded { swept } => assert_eq!(swept, 1),
            other => panic!("expected Faded, got {other:?}"),
        }
    }

    #[test]
    fn distill_where_filters_by_trait() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { act: buy, domain: ecommerce }
            PERCEIVE interaction FROM subject "u:X" AS { act: buy, domain: ecommerce }
            PERCEIVE interaction FROM subject "u:X" AS { act: read, domain: news }
            DISTILL subject "u:X" FROM perceptions WHERE domain "ecommerce" INTO intention_vector
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        match &results[3] {
            Ok(ExecResult::Dreamed(r)) => {
                // Only the 2 "ecommerce" domain perceptions are distilled; "news" is excluded.
                assert_eq!(r.perceptions_absorbed, 2, "WHERE filtered by trait");
            }
            other => panic!("expected Dreamed, got {other:?}"),
        }
    }

    #[test]
    fn fade_where_predicate_selects_what_to_forget() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { a: b } WITH salience 1.0 DECAYS halflife 100h
            PERCEIVE interaction FROM subject "u:X" AS { a: b } WITH salience 1.0 DECAYS halflife 100h
            FADE noise WHERE age > 3600 PRESERVING archetype_contribution
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        ex.execute(&stmts[0], 0.0).unwrap();
        ex.execute(&stmts[1], 0.0).unwrap();
        // 2h later: both have age = 7200 > 3600 ⇒ both fade despite high weight.
        match ex.execute(&stmts[2], 7200.0).unwrap() {
            ExecResult::Faded { swept } => assert_eq!(swept, 2, "WHERE by age swept them"),
            other => panic!("expected Faded, got {other:?}"),
        }
    }

    #[test]
    fn evoke_resolution_point_drops_arc() {
        // Two cycles to build an arc; then EVOKE with RESOLUTION point must return no milestones.
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { act: a }
            DISTILL subject "u:X" INTO intention_vector
            PERCEIVE interaction FROM subject "u:X" AS { act: b }
            DISTILL subject "u:X" INTO intention_vector
            EVOKE essence OF "u:X" RESOLUTION point WITHIN budget 800 tokens
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        match results.last().unwrap() {
            Ok(ExecResult::Evoked(ctx)) => assert!(ctx.arc_points.is_empty(), "point ⇒ no arc"),
            other => panic!("expected Evoked, got {other:?}"),
        }
    }

    #[test]
    fn imprint_consolidates_existing_archetype() {
        // Real IMPRINT: first an essence is distilled; then IMPRINT **anchors** it.
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { a: b }
            PERCEIVE interaction FROM subject "u:X" AS { a: b }
            DISTILL subject "u:X" INTO intention_vector
            IMPRINT archetype "u:X" FROM intention_vector RESILIENCE high
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        match results.last().unwrap() {
            Ok(ExecResult::Imprinted { archetype, .. }) => assert_eq!(archetype, "u:X"),
            other => panic!("expected Imprinted, got {other:?}"),
        }
        // IMPRINT actually changed the physics: essence gained reinforcement (no longer a no-op).
        let a = ex.runtime().long_term().get("u:X").unwrap();
        assert!(a.trace.reinforcement > 0.0, "IMPRINT consolidated the essence");
    }

    #[test]
    fn imprint_unknown_subject_fails() {
        let stmts =
            parse(r#"IMPRINT archetype "ghost" FROM intention_vector RESILIENCE high"#).unwrap();
        let mut ex = fresh();
        let r = &ex.execute_program(&stmts, 0.0)[0];
        assert!(
            matches!(r, Err(ExecError::NoSuchSubject(_))),
            "cannot imprint what was not distilled"
        );
    }

    #[test]
    fn evoke_resonating_with_focuses_on_the_matching_mode() {
        // Two behaviours with no shared tokens → two distinct modes.
        let src = r#"
            PERCEIVE interaction FROM subject "u" AS { topic: galaxies }
            PERCEIVE interaction FROM subject "u" AS { topic: galaxies }
            PERCEIVE interaction FROM subject "u" AS { flavor: cooking }
            PERCEIVE interaction FROM subject "u" AS { flavor: cooking }
            DISTILL subject "u" INTO intention_vector
        "#;
        let mut ex = fresh();
        ex.execute_program(&parse(src).unwrap(), 0.0);

        // RESONATING WITH is no longer ignored: focuses evocation on the resonating mode.
        let gal =
            parse(r#"EVOKE essence OF "u" RESONATING WITH { galaxies } WITHIN budget 800 tokens"#)
                .unwrap();
        match &ex.execute_program(&gal, 0.0)[0] {
            Ok(ExecResult::Evoked(ctx)) => {
                assert_eq!(ctx.resonating_mode.as_deref(), Some("galaxies"))
            }
            other => panic!("expected Evoked, got {other:?}"),
        }
        let cook =
            parse(r#"EVOKE essence OF "u" RESONATING WITH { cooking } WITHIN budget 800 tokens"#)
                .unwrap();
        match &ex.execute_program(&cook, 0.0)[0] {
            Ok(ExecResult::Evoked(ctx)) => {
                assert_eq!(ctx.resonating_mode.as_deref(), Some("cooking"))
            }
            other => panic!("expected Evoked, got {other:?}"),
        }
    }

    #[test]
    fn recall_returns_the_matching_fact_verbatim() {
        let day = 86_400.0;
        let mut ex = fresh();
        remember_fact(
            &mut ex,
            "u",
            "allergic to peanuts",
            "health allergy peanuts",
            day,
            0.0,
        );
        remember_fact(
            &mut ex,
            "u",
            "drives a red car",
            "vehicle car red",
            day,
            0.0,
        );
        let prog = parse(
            r#"RECALL facts FROM subject "u" RESONATING WITH { health, allergy } WITHIN k 1"#,
        )
        .unwrap();
        match &ex.execute_program(&prog, 0.0)[0] {
            Ok(ExecResult::Recalled(facts)) => {
                assert_eq!(facts.len(), 1);
                assert_eq!(
                    facts[0].text, "allergic to peanuts",
                    "layer-1: lossless verbatim retrieval"
                );
            }
            other => panic!("expected Recalled, got {other:?}"),
        }
    }

    #[test]
    fn reinforce_resets_decay_so_the_fact_survives() {
        let half = 30.0 * 86_400.0;
        let mut ex = fresh();
        remember_fact(&mut ex, "u", "fact alpha", "topic alpha", half, 0.0);
        remember_fact(&mut ex, "u", "fact beta", "topic beta", half, 0.0);
        // At t=half, REINFORCE only alpha → its decay is reset.
        let prog =
            parse(r#"REINFORCE facts FROM subject "u" RESONATING WITH { alpha } WITHIN k 1"#)
                .unwrap();
        match ex.execute(&prog[0], half).unwrap() {
            ExecResult::Reinforced { count } => assert_eq!(count, 1),
            other => panic!("expected Reinforced, got {other:?}"),
        }
        // Much later we sweep: reinforced survives; the other (never touched) fades.
        let swept = ex.runtime_mut().fade_facts(half * 5.0);
        assert_eq!(swept, 1, "only the unreinforced fact fades");
    }

    #[test]
    fn recall_where_resonates_filters_by_threshold() {
        let day = 86_400.0;
        let mut ex = fresh();
        remember_fact(&mut ex, "u", "strong match", "query topic here", day, 0.0);
        remember_fact(&mut ex, "u", "weak match", "query", day, 0.0);
        // Weak match resonates ~0.58 with the query (only shares "query"); threshold 0.6 drops it.
        let prog = parse(r#"RECALL facts FROM subject "u" RESONATING WITH { query, topic, here } WHERE resonates > 0.6 WITHIN k 5"#).unwrap();
        match &ex.execute_program(&prog, 0.0)[0] {
            Ok(ExecResult::Recalled(facts)) => {
                assert_eq!(facts.len(), 1, "vector predicate filtered the weak match");
                assert_eq!(facts[0].text, "strong match");
            }
            other => panic!("expected Recalled, got {other:?}"),
        }
    }
}
