//! Perception layer — volatile short-term memory.
//!
//! A `Perception` is a raw stimulus just assimilated (`PERCEIVE`). It is born decaying: its
//! `EntropyTrace` determines how much it weighs at any instant. Deliberately fragile — if nothing
//! reinforces it, it falls below `θ_fade` and the semantic GC sweeps it away.

use crate::entropy::{EntropyTrace, Tick};
use crate::vector::Vector;
use std::collections::HashMap;

/// A perceived stimulus: the semantic embedding + its traits + its entropy trace.
#[derive(Debug, Clone)]
pub struct Perception {
    /// Subject it belongs to, e.g. "user:Xolotl".
    pub subject: String,
    /// Semantic embedding of the stimulus (from the inference Provider).
    pub embedding: Vector,
    /// Raw traits (act, object, hue, urgency...). Not a fixed schema.
    pub traits: HashMap<String, String>,
    /// Forgetting physics of this stimulus.
    pub trace: EntropyTrace,
}

impl Perception {
    pub fn new(
        subject: impl Into<String>,
        embedding: Vector,
        salience: f64,
        halflife: f64,
        now: Tick,
    ) -> Self {
        Self {
            subject: subject.into(),
            embedding,
            traits: HashMap::new(),
            trace: EntropyTrace::new(salience, halflife, now),
        }
    }

    pub fn with_trait(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.traits.insert(key.into(), value.into());
        self
    }

    /// Current weight (lazy). Shortcut over the entropy trace.
    #[inline]
    pub fn weight(&self, now: Tick) -> f64 {
        self.trace.weight(now)
    }

    /// Representative text of the stimulus: the trait **values** in stable key order. This is the
    /// lexical label that survives distillation so the prose can name the content (not just vectors).
    /// E.g. `{act: purchase, object: shoes}` → "purchase shoes".
    pub fn representative_text(&self) -> String {
        let mut keys: Vec<&String> = self.traits.keys().collect();
        keys.sort();
        keys.iter()
            .map(|k| self.traits[*k].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// The short-term sensory memory: live perceptions, not yet swept by FADE.
#[derive(Debug, Default)]
pub struct PerceptionBuffer {
    perceptions: Vec<Perception>,
}

impl PerceptionBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `PERCEIVE`: assimilate a raw stimulus.
    pub fn perceive(&mut self, p: Perception) {
        self.perceptions.push(p);
    }

    pub fn len(&self) -> usize {
        self.perceptions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perceptions.is_empty()
    }

    /// Live perceptions of a subject (weight ≥ θ_fade) at `now`. Lazy: evaluates the weight here, not per tick.
    pub fn alive_for<'a>(
        &'a self,
        subject: &'a str,
        now: Tick,
        theta_fade: f64,
    ) -> impl Iterator<Item = &'a Perception> + 'a {
        self.perceptions
            .iter()
            .filter(move |p| p.subject == subject && p.weight(now) >= theta_fade)
    }

    /// Like [`alive_for`](Self::alive_for) but also requires the user's predicate (the `WHERE` clause
    /// of `DISTILL`) to hold. The predicate is evaluated outside the core, keeping it decoupled from
    /// `letheo-mql`.
    pub fn alive_for_where<'a>(
        &'a self,
        subject: &'a str,
        now: Tick,
        theta_fade: f64,
        keep: impl Fn(&Perception) -> bool + 'a,
    ) -> impl Iterator<Item = &'a Perception> + 'a {
        self.perceptions
            .iter()
            .filter(move |p| p.subject == subject && p.weight(now) >= theta_fade && keep(p))
    }

    /// `FADE`: semantic garbage collector sweep. Removes the perceptions below threshold and returns
    /// how many faded. Their contribution to the archetype was already absorbed by DISTILL.
    pub fn fade_swept(&mut self, now: Tick, theta_fade: f64) -> usize {
        let before = self.perceptions.len();
        self.perceptions.retain(|p| p.weight(now) >= theta_fade);
        before - self.perceptions.len()
    }

    /// `FADE … WHERE`: fades the perceptions that satisfy the user's predicate (the `WHERE` clause
    /// *is* the forgetting condition). Returns how many were swept.
    pub fn fade_swept_where(&mut self, drop_if: impl Fn(&Perception) -> bool) -> usize {
        let before = self.perceptions.len();
        self.perceptions.retain(|p| !drop_if(p));
        before - self.perceptions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_DAY: f64 = 12.0 * 3600.0;

    #[test]
    fn perceive_and_count() {
        let mut buf = PerceptionBuffer::new();
        buf.perceive(Perception::new(
            "user:X",
            vec![1.0, 0.0],
            0.5,
            HALF_DAY,
            0.0,
        ));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn fade_sweep_removes_decayed_noise() {
        let mut buf = PerceptionBuffer::new();
        // Low-salience, short half-life noise.
        buf.perceive(Perception::new("user:X", vec![1.0], 0.2, HALF_DAY, 0.0));
        // Strong, persistent signal.
        buf.perceive(Perception::new(
            "user:X",
            vec![1.0],
            1.0,
            HALF_DAY * 100.0,
            0.0,
        ));

        let faded = buf.fade_swept(HALF_DAY * 5.0, crate::entropy::DEFAULT_THETA_FADE);
        assert_eq!(faded, 1, "only the noise fades");
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn alive_filters_by_subject() {
        let mut buf = PerceptionBuffer::new();
        buf.perceive(Perception::new("user:X", vec![1.0], 1.0, HALF_DAY, 0.0));
        buf.perceive(Perception::new("user:Y", vec![1.0], 1.0, HALF_DAY, 0.0));
        let n = buf
            .alive_for("user:X", 0.0, crate::entropy::DEFAULT_THETA_FADE)
            .count();
        assert_eq!(n, 1);
    }
}
