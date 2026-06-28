//! The Physics of Forgetting.
//!
//! `weight(t) = salience · e^(−λ·Δt) · (1 + reinforcement)`,  `λ = ln2 / halflife`.
//!
//! **Lazy evaluation (critical):** the weight is NOT recomputed every clock tick. Each memory stores
//! only its parameters + `last_touch`; `weight()` is a pure function evaluated on demand, only during
//! `DISTILL`, `EVOKE`, or the semantic garbage collector sweep.

/// Logical runtime clock, in seconds. Abstracted so tests stay deterministic and so the physics is
/// not tied to a wall clock (time is an entropy coefficient, not a timestamp).
pub type Tick = f64;

/// `ln(2)`, used to convert half-life ↔ decay rate.
pub const LN2: f64 = std::f64::consts::LN_2;

/// **Maximum** half-life of a memory (seconds): ~10 years. Repeated consolidation lowers λ, but never
/// below the floor derived from here — so forgetting stays possible and **nothing becomes immortal no
/// matter how often it is revisited**. Consistent with the thesis "forgetting is a feature".
pub const MAX_HALFLIFE_SECS: f64 = 10.0 * 365.0 * 86_400.0;
/// λ floor derived from [`MAX_HALFLIFE_SECS`]: consolidation cannot push λ below this.
pub const LAMBDA_FLOOR: f64 = LN2 / MAX_HALFLIFE_SECS;

/// Converts a half-life (in seconds) to the decay rate λ.
#[inline]
pub fn lambda_from_halflife(halflife: f64) -> f64 {
    assert!(halflife > 0.0, "halflife must be > 0");
    LN2 / halflife
}

/// Converts λ back to a half-life (seconds).
#[inline]
pub fn halflife_from_lambda(lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be > 0");
    LN2 / lambda
}

/// A memory's entropy trace: everything needed to evaluate its weight lazily.
///
/// It does not store the weight; it *derives* it. This is what allows holding millions of memories
/// without recomputing `e^x` per tick.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropyTrace {
    /// Initial charge of the stimulus when perceived, typically `(0, 1]`.
    pub salience: f64,
    /// Decay rate λ. Can be lowered by repetition (consolidation).
    pub lambda: f64,
    /// Accumulated reinforcements (each EVOKE/repetition adds).
    pub reinforcement: f64,
    /// Tick of the last reinforcement/evocation. Δt = now − last_touch.
    pub last_touch: Tick,
}

impl EntropyTrace {
    /// Creates a fresh trace from salience and half-life (seconds), touched at `now`.
    pub fn new(salience: f64, halflife: f64, now: Tick) -> Self {
        Self {
            salience,
            lambda: lambda_from_halflife(halflife),
            reinforcement: 0.0,
            last_touch: now,
        }
    }

    /// Time elapsed since the last touch. Clamped to `≥ 0` (a clock running backwards does not
    /// "un-forget").
    #[inline]
    pub fn delta_t(&self, now: Tick) -> f64 {
        (now - self.last_touch).max(0.0)
    }

    /// The memory's weight at `now`. Pure function: mutates nothing (lazy).
    ///
    /// `weight = salience · e^(−λ·Δt) · (1 + reinforcement)`
    pub fn weight(&self, now: Tick) -> f64 {
        let decay = (-self.lambda * self.delta_t(now)).exp();
        self.salience * decay * (1.0 + self.reinforcement)
    }

    /// Reinforcement (when evoked or repeated): resets Δt to 0 and adds reinforcement with
    /// **diminishing returns** (`+= 1/(1+reinforcement)` → grows ~√n, not linearly: usage frequency no
    /// longer dominates the ranking without a ceiling; the 1st reinforcement still adds 1.0).
    /// Optionally consolidates by lowering λ, but **never below [`LAMBDA_FLOOR`]** (nothing becomes
    /// immortal).
    ///
    /// `consolidation` in `[0, 1)`: fraction by which λ is lowered by this reinforcement (0 = no change).
    pub fn reinforce(&mut self, now: Tick, consolidation: f64) {
        debug_assert!((0.0..1.0).contains(&consolidation));
        self.last_touch = now;
        self.reinforcement += 1.0 / (1.0 + self.reinforcement);
        self.lambda = (self.lambda * (1.0 - consolidation)).max(LAMBDA_FLOOR);
    }

    /// Is the memory below the forgetting threshold? (FADE candidate).
    #[inline]
    pub fn is_faded(&self, now: Tick, theta_fade: f64) -> bool {
        self.weight(now) < theta_fade
    }
}

/// Default threshold below which a memory becomes a `FADE` candidate. Calibratable per domain.
pub const DEFAULT_THETA_FADE: f64 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_DAY: f64 = 12.0 * 3600.0;

    #[test]
    fn lambda_halflife_roundtrip() {
        let l = lambda_from_halflife(HALF_DAY);
        assert!((halflife_from_lambda(l) - HALF_DAY).abs() < 1e-6);
    }

    #[test]
    fn weight_halves_after_one_halflife() {
        let t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        let w0 = t.weight(0.0);
        let w1 = t.weight(HALF_DAY);
        assert!((w0 - 1.0).abs() < 1e-9);
        assert!(
            (w1 - 0.5).abs() < 1e-6,
            "after one half-life the weight halves"
        );
    }

    #[test]
    fn weight_is_monotonic_decreasing_in_time() {
        let t = EntropyTrace::new(0.8, HALF_DAY, 0.0);
        let mut prev = f64::INFINITY;
        for step in 0..50 {
            let w = t.weight(step as f64 * 3600.0);
            assert!(w < prev, "weight must decrease monotonically with Δt");
            prev = w;
        }
    }

    #[test]
    fn weight_is_asymptotic_never_zero() {
        let t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        let w = t.weight(HALF_DAY * 1000.0);
        assert!(
            w > 0.0,
            "decay is asymptotic: it never reaches exactly 0"
        );
        assert!(w < 1e-6);
    }

    #[test]
    fn reinforcement_resets_delta_t_and_raises_weight() {
        let mut t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        // Let it decay one half-life: weight ~0.5.
        let decayed = t.weight(HALF_DAY);
        assert!((decayed - 0.5).abs() < 1e-6);
        // Evoke at that moment: Δt→0 and reinforcement→1.
        t.reinforce(HALF_DAY, 0.0);
        let after = t.weight(HALF_DAY);
        assert!(after > decayed, "remembering reinforces the weight");
        // salience(1) · e^0 · (1 + 1) = 2.0
        assert!((after - 2.0).abs() < 1e-6);
    }

    #[test]
    fn consolidation_extends_halflife() {
        let mut t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        let lambda_before = t.lambda;
        t.reinforce(0.0, 0.5); // halves λ → half-life ×2
        assert!((t.lambda - lambda_before * 0.5).abs() < 1e-12);
        assert!(
            t.lambda < lambda_before,
            "consolidation extends the half-life"
        );
    }

    #[test]
    fn reinforcement_has_diminishing_returns_and_lambda_floor() {
        // The 1st reinforcement adds 1.0 (compatibility), the rest add less each time (~√n, not
        // linear): usage frequency no longer dominates the weight without a ceiling.
        let mut t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        t.reinforce(0.0, 0.0);
        assert!((t.reinforcement - 1.0).abs() < 1e-9, "1st reinforcement = +1.0");
        for _ in 0..50 {
            t.reinforce(0.0, 0.0);
        }
        assert!(
            t.reinforcement < 12.0,
            "51 reinforcements ⇒ ≈√n, not 51: {}",
            t.reinforcement
        );
        assert!(t.reinforcement > 1.0, "but it keeps growing");

        // λ floor: no matter how much you consolidate, the half-life never becomes infinite (no immortality).
        let mut c = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        for _ in 0..500 {
            c.reinforce(0.0, 0.5);
        }
        assert!(
            (c.lambda - LAMBDA_FLOOR).abs() < 1e-18,
            "λ bottoms out at the floor, not at 0"
        );
    }

    #[test]
    fn fade_triggers_below_threshold() {
        let t = EntropyTrace::new(0.2, HALF_DAY, 0.0);
        assert!(
            !t.is_faded(0.0, DEFAULT_THETA_FADE),
            "just perceived does not fade"
        );
        // After enough time it falls below θ_fade.
        assert!(t.is_faded(HALF_DAY * 5.0, DEFAULT_THETA_FADE));
    }
}
