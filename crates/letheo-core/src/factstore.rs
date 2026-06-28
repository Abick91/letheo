//! Fact Layer — **episodic** memory (layer-1, the hippocampus).
//!
//! The archetype (layer-2, neocortex) *generalises*: compresses behaviour into modes and forgets
//! redundancy. But there are things an agent cannot afford to average — "allergic to peanuts",
//! "the car is the red one", "already delivered milestone 3". That is **verbatim episodic
//! knowledge**: specific, lossless, recoverable word-for-word.
//!
//! Until now that layer lived **outside the engine**, as a Python list in the consumer layer:
//! no forgetting physics, no deduplication, no index. Here it enters the core under the **same
//! physics** as everything else ([`EntropyTrace`]). It is the *Complementary Learning Systems*
//! model made literal: two representations (fast episodic ↔ slow semantic), **one single decay
//! physics**. A fact not revisited decays and the semantic GC sweeps it; one recalled or repeated
//! is reinforced (spaced repetition). There are no two systems taped together: there is one engine
//! with two layers.

use crate::entropy::{EntropyTrace, Tick};
use crate::vector::{cosine, Vector};

/// Cosine threshold above which two facts are considered **the same fact** (dedup → reinforcement
/// instead of insertion). High on purpose: layer-1 is verbatim, so only near-identical repetitions
/// collapse; a distant paraphrase is stored as a distinct fact. Declared physics, adjustable via
/// [`FactStore::with_dedup`] (not a magic constant — consistent with TRUTH 100% discipline).
pub const DEFAULT_FACT_DEDUP: f32 = 0.95;

/// Consolidation on reinforcement of a fact (repetition or evocation): fraction by which λ is
/// reduced, extending the half-life (spaced repetition / FSRS). Small on purpose: recalling a
/// fact anchors it, but forgetting remains real if it is not touched again.
pub const DEFAULT_FACT_CONSOLIDATION: f64 = 0.1;

/// An episodic fact: **verbatim** content + its embedding + its forgetting physics.
#[derive(Debug, Clone)]
pub struct Fact {
    /// Subject the fact belongs to (e.g. "user:Xolotl", "agent:adder").
    pub subject: String,
    /// Exact content, lossless. This is the layer-1 promise: returned exactly as learned.
    pub text: String,
    /// Semantic embedding of the fact (from the inference Provider). Basis for retrieval and dedup.
    pub embedding: Vector,
    /// Provenance: which agent/source contributed the fact. For auditing and the memory market (L12).
    pub provenance: String,
    /// Tick when the fact was learned (creation). Immutable; distinct from `trace.last_touch` (reinforcement).
    pub created_at: Tick,
    /// Forgetting physics of the fact: decays unless reinforced/evoked. The same [`EntropyTrace`]
    /// that governs perceptions, modes and archetypes — one single physics for both layers.
    pub trace: EntropyTrace,
}

/// Result of `remember`: whether the fact was new or collapsed (reinforcement) onto an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Remember {
    /// New fact: inserted into the store.
    Inserted,
    /// Already known fact (≥ `theta_dedup`): not duplicated, the existing one was reinforced.
    Merged,
}

/// A fact retrieved by `recall`: its verbatim text + provenance + the physical score it ranked with.
/// A decoupled clone from the store (the `recall` reinforced the original fact when evoked).
#[derive(Debug, Clone)]
pub struct RecalledFact {
    pub text: String,
    pub provenance: String,
    /// `relevance · life` score used for ranking (see [`FactStore::recall`]).
    pub score: f64,
}

/// Episodic memory: verbatim facts with forgetting, deduplication and index (Flat for now; the ANN
/// from L3 plugs in here without changing semantics). Multi-subject: dedup and recall are isolated
/// per subject, like [`crate::perception::PerceptionBuffer`].
#[derive(Debug, Clone)]
pub struct FactStore {
    facts: Vec<Fact>,
    theta_dedup: f32,
}

impl Default for FactStore {
    fn default() -> Self {
        Self {
            facts: Vec::new(),
            theta_dedup: DEFAULT_FACT_DEDUP,
        }
    }
}

impl FactStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a store with an explicit dedup threshold (see [`DEFAULT_FACT_DEDUP`]).
    pub fn with_dedup(theta_dedup: f32) -> Self {
        Self {
            facts: Vec::new(),
            theta_dedup,
        }
    }

    pub fn theta_dedup(&self) -> f32 {
        self.theta_dedup
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Iterates facts in insertion order (for snapshots / persistence).
    pub fn iter(&self) -> impl Iterator<Item = &Fact> {
        self.facts.iter()
    }

    /// Inserts an already-built fact **without** dedup or new physics (e.g. when rehydrating from disk).
    /// To register new knowledge use [`remember`](Self::remember).
    pub fn insert(&mut self, fact: Fact) {
        self.facts.push(fact);
    }

    /// Live facts for a subject (weight ≥ θ_fade) at `now`. Lazy: evaluates weight here, not per tick.
    pub fn alive_for<'a>(
        &'a self,
        subject: &'a str,
        now: Tick,
        theta_fade: f64,
    ) -> impl Iterator<Item = &'a Fact> + 'a {
        self.facts
            .iter()
            .filter(move |f| f.subject == subject && f.trace.weight(now) >= theta_fade)
    }

    /// Records a fact. If one already exists for the **same subject** whose direction resonates ≥ `theta_dedup`,
    /// it is not duplicated: the existing one is **reinforced** (repetition consolidates, does not inflate
    /// the store). Otherwise it is inserted with its own forgetting physics (`salience`, `halflife`).
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
        let subject = subject.into();
        // Dedup within the subject: the most similar fact, if above threshold, is "the same fact".
        let mut best = self.theta_dedup;
        let mut best_i: Option<usize> = None;
        for (i, f) in self.facts.iter().enumerate() {
            if f.subject != subject {
                continue;
            }
            let c = cosine(&f.embedding, &embedding);
            if c >= best {
                best = c;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => {
                self.facts[i]
                    .trace
                    .reinforce(now, DEFAULT_FACT_CONSOLIDATION);
                Remember::Merged
            }
            None => {
                self.facts.push(Fact {
                    subject,
                    text: text.into(),
                    embedding,
                    provenance: provenance.into(),
                    created_at: now,
                    trace: EntropyTrace::new(salience, halflife, now),
                });
                Remember::Inserted
            }
        }
    }

    /// **Directed retrieval** (layer-1): the `k` facts for the subject that score best against the
    /// query, ranked by native physics `score = max(0, relevance) · weight(now)` — the same form as
    /// L2, not an additive patch. Recalling **reinforces** the returned facts (spaced repetition: a
    /// recalled fact resets its decay and survives; one never evoked fades). Returns decoupled verbatim
    /// copies ([`RecalledFact`]).
    pub fn recall(
        &mut self,
        subject: &str,
        query: &[f32],
        k: usize,
        now: Tick,
        theta_fade: f64,
    ) -> Vec<RecalledFact> {
        let mut scored: Vec<(f64, usize)> = self
            .facts
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                if f.subject != subject {
                    return None;
                }
                let life = f.trace.weight(now); // `e^x` once per fact (was 2×)
                if life < theta_fade {
                    return None;
                }
                let relevance = cosine(&f.embedding, query).max(0.0) as f64;
                Some((relevance * life, i))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        scored
            .into_iter()
            .map(|(score, i)| {
                // Recalling is touching: the retrieved fact is reinforced (Δt→0) → its forgetting is deferred.
                self.facts[i]
                    .trace
                    .reinforce(now, DEFAULT_FACT_CONSOLIDATION);
                let f = &self.facts[i];
                RecalledFact {
                    text: f.text.clone(),
                    provenance: f.provenance.clone(),
                    score,
                }
            })
            .collect()
    }

    /// **Read-only** search using the same physics as `recall`, without reinforcing (for inspection/tests
    /// and to compose a unified EVOKE without side effects). Returns `(score, &Fact)` top-`k`.
    pub fn search<'a>(
        &'a self,
        subject: &str,
        query: &[f32],
        k: usize,
        now: Tick,
        theta_fade: f64,
    ) -> Vec<(f64, &'a Fact)> {
        let mut scored: Vec<(f64, &Fact)> = self
            .facts
            .iter()
            .filter_map(|f| {
                if f.subject != subject {
                    return None;
                }
                let life = f.trace.weight(now); // `e^x` once per fact (was 2×)
                if life < theta_fade {
                    return None;
                }
                let relevance = cosine(&f.embedding, query).max(0.0) as f64;
                Some((relevance * life, f))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// `FADE` of the episodic layer: sweeps facts below threshold (their life dropped below θ_fade) and
    /// returns how many faded. Forgetting is real even for exact facts.
    pub fn fade(&mut self, now: Tick, theta_fade: f64) -> usize {
        let before = self.facts.len();
        self.facts.retain(|f| f.trace.weight(now) >= theta_fade);
        before - self.facts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::DEFAULT_THETA_FADE;

    const DAY: f64 = 86_400.0;

    #[test]
    fn dedup_collapses_near_identical_facts_and_reinforces() {
        let mut fs = FactStore::new();
        assert_eq!(
            fs.remember(
                "u",
                "bought running shoes",
                vec![1.0, 0.0],
                "agentA",
                1.0,
                DAY,
                0.0
            ),
            Remember::Inserted
        );
        // Same fact again (identical embedding): not duplicated, reinforced.
        assert_eq!(
            fs.remember(
                "u",
                "bought running shoes",
                vec![1.0, 0.0],
                "agentA",
                1.0,
                DAY,
                1.0
            ),
            Remember::Merged
        );
        assert_eq!(fs.len(), 1, "the repeated fact does not inflate the store");
        assert!(
            fs.iter().next().unwrap().trace.reinforcement > 0.0,
            "repetition consolidates"
        );
    }

    #[test]
    fn distinct_facts_are_kept_separate() {
        let mut fs = FactStore::new();
        fs.remember("u", "loves noir films", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        fs.remember(
            "u",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "a",
            1.0,
            DAY,
            0.0,
        ); // orthogonal
        assert_eq!(fs.len(), 2, "distinct facts do not collapse");
    }

    #[test]
    fn recall_returns_the_relevant_fact_verbatim() {
        let mut fs = FactStore::new();
        fs.remember(
            "u",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "a",
            1.0,
            DAY,
            0.0,
        );
        fs.remember("u", "drives a red car", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        let hits = fs.recall("u", &[0.0, 1.0], 1, 0.0, DEFAULT_THETA_FADE);
        assert_eq!(hits.len(), 1);
        // The layer-1 promise: the exact fact, word for word (not a gist).
        assert_eq!(hits[0].text, "allergic to peanuts");
    }

    #[test]
    fn recall_ranks_fresh_over_stale_at_equal_relevance() {
        let half = 30.0 * DAY;
        let mut fs = FactStore::new();
        // Nearly identical directions but below the dedup threshold (cos≈0.92 < 0.95) → two facts.
        fs.remember("u", "stale fact", vec![1.0, 0.0], "a", 1.0, half, 0.0);
        fs.remember("u", "fresh fact", vec![0.92, 0.392], "a", 1.0, half, half);
        let q = [0.96, 0.2]; // halfway: comparable relevance for both
        let hits = fs.search("u", &q, 2, half, DEFAULT_THETA_FADE);
        assert_eq!(hits.len(), 2, "both still alive");
        assert_eq!(
            hits[0].1.text, "fresh fact",
            "at comparable relevance, the more alive fact comes first"
        );
    }

    #[test]
    fn recalling_a_fact_resets_its_decay() {
        let half = 30.0 * DAY;
        let mut fs = FactStore::new();
        fs.remember("u", "recalled fact", vec![1.0, 0.0], "a", 1.0, half, 0.0);
        fs.remember("u", "forgotten fact", vec![0.0, 1.0], "a", 1.0, half, 0.0);
        // At t=half we recall only the first (the query points to it): reinforced → Δt→0.
        let hits = fs.recall("u", &[1.0, 0.0], 1, half, DEFAULT_THETA_FADE);
        assert_eq!(hits[0].text, "recalled fact");
        // Much later we sweep: the recalled fact survives (clock reset), the other fades.
        let later = half * 5.0;
        let faded = fs.fade(later, DEFAULT_THETA_FADE);
        assert_eq!(faded, 1, "only the fact that was never recalled fades");
        assert_eq!(fs.len(), 1);
        assert_eq!(fs.iter().next().unwrap().text, "recalled fact");
    }

    #[test]
    fn fade_sweeps_decayed_facts() {
        let mut fs = FactStore::new();
        fs.remember("u", "fleeting", vec![1.0, 0.0], "a", 0.2, DAY, 0.0); // low salience, short life
        fs.remember("u", "durable", vec![0.0, 1.0], "a", 1.0, DAY * 100.0, 0.0);
        let faded = fs.fade(DAY * 5.0, DEFAULT_THETA_FADE);
        assert_eq!(faded, 1, "only the fragile fact fades");
        assert_eq!(fs.iter().next().unwrap().text, "durable");
    }

    #[test]
    fn recall_isolates_subjects() {
        let mut fs = FactStore::new();
        fs.remember("alice", "alice secret", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        fs.remember("bob", "bob secret", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        // Same direction but different subject ⇒ NOT deduplicated (dedup is per-subject).
        assert_eq!(fs.len(), 2);
        let hits = fs.recall("alice", &[1.0, 0.0], 5, 0.0, DEFAULT_THETA_FADE);
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].text, "alice secret",
            "facts from one subject do not leak to another"
        );
    }
}
