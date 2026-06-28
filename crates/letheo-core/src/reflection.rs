//! Reflection layer — generative memory (L8) and predictive compression (L9).
//!
//! "**Intelligence = compression**": better memory **predicts the future better** from compressed past.
//! Two capabilities, both deterministic and **LLM-free** (reflection is structural arc analysis,
//! not prose generation):
//!
//! - **L9 · predictive compression** ([`predictive_compression`]): trains the essence on the past and
//!   measures how well the held-out future resonates with it. If **modes** predict better than the single
//!   centroid, the multi-modal decomposition *understands* the behaviour, not just describes it.
//! - **L8 · reflection** ([`reflect`]): synthesises **insights** absent from any individual event
//!   —dominant transitions between cycles and *revivals* (a behaviour that peaked, fell, and returned)—
//!   derived from the trajectory. It is the "sleep reflection" made structure.

use crate::archetype::{ArcMilestone, Archetype};
use crate::perception::Perception;
use crate::synthesis::{distill, DistillConfig};
use crate::vector::{cosine, Vector};
use std::collections::HashMap;

/// Default salience of an insight materialised as a fact: high (it is wisdom distilled from the arc,
/// not a raw event). Declared, adjustable.
pub const DEFAULT_INSIGHT_SALIENCE: f64 = 0.9;

// ─────────────────────────────────────────────────────────────────────────────
// L9 — Predictive compression (the internal north-star metric)
// ─────────────────────────────────────────────────────────────────────────────

/// How well the essence predicts held-out future behaviour, broken down by representation:
/// **modes** (layer-2 multi-modal) vs the single **centroid** (blind average).
/// `modal > centroid` ⇒ the multi-modal decomposition provides real predictive power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictiveScore {
    /// Mean resonance (cosine, ≥0) of held-out events with the best mode.
    pub modal: f64,
    /// Mean resonance of held-out events with the single centroid.
    pub centroid: f64,
    /// How many held-out events were evaluated.
    pub held_out: usize,
}

/// **L9** — trains the essence on `events[..k]` (k = `train_frac`) and measures the mean resonance of
/// the remaining (held-out) events with modes and the centroid. Deterministic. `None` if insufficient
/// data (< 2 events) or dimensions do not allow distillation.
pub fn predictive_compression(
    events: &[&Perception],
    train_frac: f64,
    cfg: DistillConfig,
) -> Option<PredictiveScore> {
    if events.len() < 2 {
        return None;
    }
    let frac = train_frac.clamp(0.0, 1.0);
    let n_train = ((events.len() as f64 * frac).round() as usize).clamp(1, events.len() - 1);
    let (train, test) = events.split_at(n_train);
    let iv = distill("predict", train, cfg)?;

    let mut modal_sum = 0.0;
    let mut centroid_sum = 0.0;
    for e in test {
        let centroid_res = cosine(&iv.centroid, &e.embedding).max(0.0) as f64;
        let modal_res = if iv.modes.is_empty() {
            centroid_res
        } else {
            iv.modes
                .iter()
                .map(|m| cosine(&m.centroid, &e.embedding))
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0) as f64
        };
        modal_sum += modal_res;
        centroid_sum += centroid_res;
    }
    let n = test.len() as f64;
    Some(PredictiveScore {
        modal: modal_sum / n,
        centroid: centroid_sum / n,
        held_out: test.len(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// L8 — Reflection (higher-order insights)
// ─────────────────────────────────────────────────────────────────────────────

/// Prevalence (fraction of peak) below which a domain is considered "decayed", for revival detection.
/// Declared physics, adjustable.
pub const DEFAULT_REVIVAL_FLOOR: f32 = 0.25;

/// A higher-order insight: a statement **absent from any individual event**, derived from the subject's
/// trajectory (arc).
#[derive(Debug, Clone, PartialEq)]
pub enum Insight {
    /// The subject tends to move from `from` to `to` (dominant transition between consecutive cycles).
    Transition {
        from: String,
        to: String,
        support: usize,
    },
    /// A behaviour that peaked, fell below the floor, and **came back** above it.
    Revival { domain: String },
}

/// **L8** — reflects over the arc: synthesises the dominant transition between consecutive cycle labels
/// and revivals. Deterministic, LLM-free. Empty if the arc is too short.
pub fn reflect(arc: &[ArcMilestone]) -> Vec<Insight> {
    let mut out = Vec::new();

    // Dominant transition: the most frequent (label_i → label_{i+1}) pair with distinct labels.
    let mut trans: HashMap<(String, String), usize> = HashMap::new();
    for w in arc.windows(2) {
        let (a, b) = (&w[0].label, &w[1].label);
        if a != b && !a.is_empty() && !b.is_empty() {
            *trans.entry((a.clone(), b.clone())).or_insert(0) += 1;
        }
    }
    // Deterministic tiebreak: higher support, then alphabetical order of (from, to).
    if let Some(((from, to), support)) = trans
        .into_iter()
        .max_by(|x, y| x.1.cmp(&y.1).then_with(|| (y.0).cmp(&x.0)))
    {
        out.push(Insight::Transition { from, to, support });
    }

    out.extend(detect_revivals(arc, DEFAULT_REVIVAL_FLOOR));
    out
}

/// Predicts the next behaviour given the current one, using transitions learned from the arc. Basis
/// of L8 verification: transition prediction must beat predicting the global arc mode.
pub fn predict_next(arc: &[ArcMilestone], current: &str) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for w in arc.windows(2) {
        if w[0].label == current && w[1].label != current {
            *counts.entry(w[1].label.as_str()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by(|x, y| x.1.cmp(&y.1).then_with(|| (y.0).cmp(x.0)))
        .map(|(l, _)| l.to_string())
}

/// Materialises an insight as `(text, embedding)` for storage as a **high-salience fact** in layer-1.
/// The embedding is derived from the referenced behaviour (mode centroid, or milestone direction)
/// **without a provider** — it is geometry the engine already has. A `RECALL` that resonates with
/// that behaviour will therefore also retrieve the insight. `None` if the referenced behaviour has no
/// known direction in the archetype.
pub fn materialize(archetype: &Archetype, insight: &Insight) -> Option<(String, Vector)> {
    let (text, label) = match insight {
        Insight::Transition { from, to, .. } => (
            format!("behaviour transition: {from} → {to}"),
            to.as_str(),
        ),
        Insight::Revival { domain } => (
            format!("recurring behaviour (revival): {domain}"),
            domain.as_str(),
        ),
    };
    let dir = direction_for_label(archetype, label)?;
    Some((text, dir))
}

/// Direction (embedding) associated with a label: its mode centroid if it exists, or the direction of
/// the most recent milestone with that label. Geometry already present in the archetype, no provider.
fn direction_for_label(a: &Archetype, label: &str) -> Option<Vector> {
    a.modes
        .iter()
        .find(|m| m.label == label)
        .map(|m| m.centroid.clone())
        .or_else(|| {
            a.arc
                .iter()
                .rev()
                .find(|m| m.label == label)
                .map(|m| m.direction.clone())
        })
}

/// Detects revivals: for each domain, its prevalence per milestone (from `label_histogram`) shows a
/// peak, then a drop below `floor·peak`, and then a rebound above `floor·peak`.
fn detect_revivals(arc: &[ArcMilestone], floor: f32) -> Vec<Insight> {
    if arc.len() < 3 {
        return Vec::new();
    }
    let totals: Vec<usize> = arc
        .iter()
        .map(|m| m.label_histogram.iter().map(|(_, c)| c).sum())
        .collect();

    // Universe of domains, in deterministic order.
    let mut domains: Vec<&str> = arc
        .iter()
        .flat_map(|m| m.label_histogram.iter().map(|(l, _)| l.as_str()))
        .collect();
    domains.sort_unstable();
    domains.dedup();

    let mut revivals = Vec::new();
    for d in domains {
        let series: Vec<f32> = arc
            .iter()
            .zip(&totals)
            .map(|(m, &t)| {
                let c = m
                    .label_histogram
                    .iter()
                    .find(|(l, _)| l == d)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);
                if t > 0 {
                    c as f32 / t as f32
                } else {
                    0.0
                }
            })
            .collect();
        let peak = series.iter().cloned().fold(0.0_f32, f32::max);
        if peak <= 0.0 {
            continue;
        }
        let thr = floor * peak;
        let peak_i = series
            .iter()
            .position(|&x| (x - peak).abs() < 1e-6)
            .unwrap();
        // After the peak: is there a drop below the floor and then a rebound above it?
        let (mut dipped, mut revived) = (false, false);
        for &x in &series[peak_i + 1..] {
            if x < thr {
                dipped = true;
            } else if dipped && x >= thr {
                revived = true;
                break;
            }
        }
        if dipped && revived {
            revivals.push(Insight::Revival {
                domain: d.to_string(),
            });
        }
    }
    revivals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(act: &str, e: Vec<f32>) -> Perception {
        Perception::new("u", e, 1.0, 3600.0, 0.0).with_trait("act", act)
    }

    fn milestone(label: &str, hist: &[(&str, usize)]) -> ArcMilestone {
        ArcMilestone {
            at: 0.0,
            direction: vec![1.0, 0.0],
            absorbed: hist.iter().map(|(_, c)| c).sum(),
            label: label.to_string(),
            label_histogram: hist.iter().map(|(l, c)| (l.to_string(), *c)).collect(),
        }
    }

    fn arc_of(labels: &[&str]) -> Vec<ArcMilestone> {
        labels
            .iter()
            .enumerate()
            .map(|(i, l)| ArcMilestone {
                at: i as f64,
                direction: vec![1.0, 0.0],
                absorbed: 1,
                label: l.to_string(),
                label_histogram: vec![(l.to_string(), 1)],
            })
            .collect()
    }

    #[test]
    fn modes_predict_held_out_better_than_centroid() {
        // Bimodal behaviour (A and B orthogonal, alternating): the mean falls between both and predicts
        // worse than recovering the correct mode for each future event.
        let mut ps = Vec::new();
        for _ in 0..20 {
            ps.push(p("A", vec![1.0, 0.0]));
            ps.push(p("B", vec![0.0, 1.0]));
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let s = predictive_compression(&refs, 0.7, DistillConfig::default()).unwrap();
        assert!(s.held_out > 0);
        assert!(
            s.modal > s.centroid + 0.2,
            "modes predict the future better than the mean: {s:?}"
        );
    }

    #[test]
    fn unimodal_modes_match_centroid() {
        // Single behaviour: mode ≈ centroid → no predictive gain (or loss).
        let ps: Vec<Perception> = (0..20)
            .map(|i| p("x", vec![1.0, 0.01 * i as f32]))
            .collect();
        let refs: Vec<&Perception> = ps.iter().collect();
        let s = predictive_compression(&refs, 0.7, DistillConfig::default()).unwrap();
        assert!(
            (s.modal - s.centroid).abs() < 0.1,
            "unimodal: modes ≈ centroid: {s:?}"
        );
    }

    #[test]
    fn predictive_compression_needs_data() {
        let one = p("a", vec![1.0, 0.0]);
        assert!(predictive_compression(&[&one], 0.7, DistillConfig::default()).is_none());
    }

    #[test]
    fn reflect_finds_dominant_transition() {
        // trail→yoga occurs twice; it is the dominant transition in the arc.
        let arc = arc_of(&["trail", "yoga", "trail", "yoga", "climb"]);
        let insights = reflect(&arc);
        assert!(
            insights
                .iter()
                .any(|i| matches!(i, Insight::Transition { from, to, support }
                if from == "trail" && to == "yoga" && *support == 2)),
            "{insights:?}"
        );
    }

    #[test]
    fn reflect_detects_revival() {
        // yoga: peak (0.8), drops (0.1), returns (0.7) → revival; trail does not revive (monotone rise).
        let arc = vec![
            milestone("yoga", &[("yoga", 8), ("trail", 2)]),
            milestone("trail", &[("yoga", 1), ("trail", 9)]),
            milestone("yoga", &[("yoga", 7), ("trail", 3)]),
        ];
        let insights = reflect(&arc);
        assert!(
            insights
                .iter()
                .any(|i| matches!(i, Insight::Revival { domain } if domain == "yoga")),
            "{insights:?}"
        );
        assert!(
            !insights
                .iter()
                .any(|i| matches!(i, Insight::Revival { domain } if domain == "trail")),
            "trail does not revive: {insights:?}"
        );
    }

    #[test]
    fn transition_prediction_beats_marginal() {
        // A→B structure: given "A", the real next is always "B". The global mode misses it;
        // transition prediction gets it right.
        let arc = arc_of(&["A", "B", "A", "B", "A", "B"]);
        assert_eq!(
            predict_next(&arc, "A").as_deref(),
            Some("B"),
            "transition predicts B after A"
        );
    }
}
