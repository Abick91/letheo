//! # letheo-async · The organism that breathes on its own
//!
//! In the core, `breathe()` is synchronous: the caller decides *when* the runtime dreams. Here we
//! complete the last architectural pillar: mounting that loop on **Tokio** so the semantic GC runs
//! **in the background**, without blocking perception and without anyone calling `breathe`.
//!
//! Design:
//! - **Separate crate**, not inside `letheo-core`: the core stays deterministic and offline
//!   (`cargo test -p letheo-core` does not pull in Tokio). All async code lives here.
//! - **Logical clock derived from wall clock**: `Tick = elapsed_seconds · time_scale`. With
//!   `tokio::time::pause()` tests control time and are deterministic despite being async.
//! - **Two sleep triggers**: a periodic `interval` (basal breathing) and *backpressure*
//!   (if ≥ `pressure_watermark` unconsolidated perceptions accumulate, dream immediately).
//!
//! The actor has exclusive ownership of `CognitiveRuntime`; the outside world communicates
//! through channels. This avoids locks and keeps the `!Sync` core frictionless.

use std::collections::HashSet;
use std::time::Duration;

use letheo_core::{
    BreathReport, CognitiveRuntime, CompressedContext, EvokeRequest, Perception, RuntimeConfig,
    Tick,
};
use letheo_inference::Provider;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{interval_at, Instant, MissedTickBehavior};

/// Configuration for the async runtime.
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Cognitive core config (thresholds, resilience, etc.).
    pub runtime: RuntimeConfig,
    /// How often the organism breathes basally.
    pub breath_interval: Duration,
    /// How many unconsolidated perceptions trigger an immediate dream (backpressure). 0 = never.
    pub pressure_watermark: usize,
    /// Logical seconds (`Tick`) per real second. Allows accelerating physics vs wall-clock.
    pub time_scale: f64,
    /// Command channel capacity (bounded perception queue).
    pub channel_capacity: usize,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            breath_interval: Duration::from_secs(60),
            pressure_watermark: 256,
            time_scale: 1.0,
            channel_capacity: 1024,
        }
    }
}

/// Cumulative metrics of the organism (observability).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Dream cycles executed (basal + pressure + on-demand).
    pub breaths: u64,
    /// Dreams triggered by the basal interval.
    pub breaths_basal: u64,
    /// Dreams triggered by backpressure.
    pub breaths_pressure: u64,
    /// Dreams triggered on-demand (`breathe(...)`).
    pub breaths_ondemand: u64,
    /// Subjects consolidated in total.
    pub distilled_subjects: u64,
    /// Perceptions absorbed into archetypes.
    pub perceptions_absorbed: u64,
    /// Perceptions swept by FADE.
    pub faded: u64,
    /// Perceptions received since the last dream (current pressure).
    pub pending: usize,
    /// Size of short-term memory.
    pub short_term_len: usize,
    /// Number of live archetypes.
    pub long_term_len: usize,
}

impl Stats {
    /// Renders metrics in Prometheus text exposition format. The output can be served directly on a
    /// `/metrics` endpoint (no dependencies or embedded server required here).
    pub fn render_prometheus(&self) -> String {
        let mut s = String::new();
        let mut metric = |name: &str, help: &str, kind: &str, value: u64| {
            s.push_str(&format!("# HELP letheo_{name} {help}\n"));
            s.push_str(&format!("# TYPE letheo_{name} {kind}\n"));
            s.push_str(&format!("letheo_{name} {value}\n"));
        };
        metric(
            "breaths_total",
            "Dream cycles executed.",
            "counter",
            self.breaths,
        );
        metric(
            "breaths_basal_total",
            "Dreams triggered by basal interval.",
            "counter",
            self.breaths_basal,
        );
        metric(
            "breaths_pressure_total",
            "Dreams triggered by backpressure.",
            "counter",
            self.breaths_pressure,
        );
        metric(
            "breaths_ondemand_total",
            "Dreams triggered on-demand.",
            "counter",
            self.breaths_ondemand,
        );
        metric(
            "distilled_subjects_total",
            "Subjects consolidated.",
            "counter",
            self.distilled_subjects,
        );
        metric(
            "perceptions_absorbed_total",
            "Perceptions absorbed into archetypes.",
            "counter",
            self.perceptions_absorbed,
        );
        metric(
            "faded_total",
            "Perceptions swept by FADE.",
            "counter",
            self.faded,
        );
        metric(
            "pending",
            "Unconsolidated perceptions (current pressure).",
            "gauge",
            self.pending as u64,
        );
        metric(
            "short_term_len",
            "Short-term memory size.",
            "gauge",
            self.short_term_len as u64,
        );
        metric(
            "long_term_len",
            "Live archetypes.",
            "gauge",
            self.long_term_len as u64,
        );
        s
    }
}

/// Source of the last dream — useful for tests and traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreathCause {
    Basal,
    Pressure,
    OnDemand,
}

/// Communication error with the actor (the organism stopped breathing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped;

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "async runtime has stopped")
    }
}
impl std::error::Error for Stopped {}

type Reply<T> = oneshot::Sender<T>;

enum Cmd {
    Perceive(Perception),
    PerceiveText {
        subject: String,
        text: String,
        salience: f64,
        halflife: f64,
    },
    Breathe {
        subjects: Option<Vec<String>>,
        reply: Reply<BreathReport>,
    },
    Evoke {
        req: EvokeRequest,
        reply: Reply<Option<CompressedContext>>,
    },
    Stats(Reply<Stats>),
}

/// Cloneable handle to the organism. All operations are async and non-blocking.
#[derive(Clone)]
pub struct AsyncRuntime {
    tx: mpsc::Sender<Cmd>,
}

impl AsyncRuntime {
    /// Starts the organism: creates a background actor with an embedding provider and a fresh core.
    /// Returns the handle and the actor's `JoinHandle` (which finishes when all handles are dropped).
    pub fn spawn<P>(provider: P, cfg: AsyncConfig) -> (Self, JoinHandle<()>)
    where
        P: Provider + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(cfg.channel_capacity);
        let actor = Actor::new(provider, cfg);
        let handle = tokio::spawn(actor.run(rx));
        (Self { tx }, handle)
    }

    /// `PERCEIVE`: assimilates an already-embedded stimulus.
    pub async fn perceive(&self, p: Perception) -> Result<(), Stopped> {
        self.tx.send(Cmd::Perceive(p)).await.map_err(|_| Stopped)
    }

    /// Ergonomic `PERCEIVE`: the actor embeds the text with its provider (traits as stimulus).
    pub async fn perceive_text(
        &self,
        subject: impl Into<String>,
        text: impl Into<String>,
        salience: f64,
        halflife: f64,
    ) -> Result<(), Stopped> {
        self.tx
            .send(Cmd::PerceiveText {
                subject: subject.into(),
                text: text.into(),
                salience,
                halflife,
            })
            .await
            .map_err(|_| Stopped)
    }

    /// Forces an on-demand dream cycle. `subjects = None` ⇒ all subjects seen so far.
    pub async fn breathe(&self, subjects: Option<Vec<String>>) -> Result<BreathReport, Stopped> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Cmd::Breathe { subjects, reply })
            .await
            .map_err(|_| Stopped)?;
        rx.await.map_err(|_| Stopped)
    }

    /// `EVOKE`: resolves the essence of a subject within the token budget.
    pub async fn evoke(&self, req: EvokeRequest) -> Result<Option<CompressedContext>, Stopped> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Cmd::Evoke { req, reply })
            .await
            .map_err(|_| Stopped)?;
        rx.await.map_err(|_| Stopped)
    }

    /// Snapshot of the organism's metrics.
    pub async fn stats(&self) -> Result<Stats, Stopped> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::Stats(reply)).await.map_err(|_| Stopped)?;
        rx.await.map_err(|_| Stopped)
    }

    /// Metrics in Prometheus exposition format, ready for a `/metrics` endpoint.
    pub async fn metrics_prometheus(&self) -> Result<String, Stopped> {
        Ok(self.stats().await?.render_prometheus())
    }
}

/// The actor: exclusive owner of the runtime. Lives in its own Tokio task.
struct Actor<P: Provider> {
    rt: CognitiveRuntime,
    provider: P,
    cfg: AsyncConfig,
    start: Instant,
    subjects: HashSet<String>,
    pending: usize,
    stats: Stats,
}

impl<P: Provider> Actor<P> {
    fn new(provider: P, cfg: AsyncConfig) -> Self {
        Self {
            rt: CognitiveRuntime::new(cfg.runtime.clone()),
            provider,
            cfg,
            start: Instant::now(),
            subjects: HashSet::new(),
            pending: 0,
            stats: Stats::default(),
        }
    }

    /// Current logical tick = elapsed real seconds · time_scale.
    fn now(&self) -> Tick {
        self.start.elapsed().as_secs_f64() * self.cfg.time_scale
    }

    async fn run(mut self, mut rx: mpsc::Receiver<Cmd>) {
        // `interval_at` with first fire at `breath_interval` (not immediate): the newborn does not
        // dream before it has perceived anything.
        let first = Instant::now() + self.cfg.breath_interval;
        let mut ticker = interval_at(first, self.cfg.breath_interval);
        // If the executor falls behind, don't accumulate a burst of dreams: skip missed ticks.
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                maybe_cmd = rx.recv() => {
                    match maybe_cmd {
                        Some(cmd) => self.handle(cmd),
                        None => break, // all handles dropped: the organism ends gracefully.
                    }
                }
                _ = ticker.tick() => {
                    self.breathe_all(BreathCause::Basal);
                }
            }
        }
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Perceive(p) => self.ingest(p),
            Cmd::PerceiveText {
                subject,
                text,
                salience,
                halflife,
            } => {
                let now = self.now();
                let embedding = self.provider.embed(&text);
                self.ingest(Perception::new(subject, embedding, salience, halflife, now));
            }
            Cmd::Breathe { subjects, reply } => {
                let report = match subjects {
                    Some(s) => self.breathe_some(&s, BreathCause::OnDemand),
                    None => self.breathe_all(BreathCause::OnDemand),
                };
                let _ = reply.send(report);
            }
            Cmd::Evoke { req, reply } => {
                let ctx = self.rt.evoke(&req, self.now());
                let _ = reply.send(ctx);
            }
            Cmd::Stats(reply) => {
                let _ = reply.send(self.snapshot());
            }
        }
    }

    fn ingest(&mut self, p: Perception) {
        self.subjects.insert(p.subject.clone());
        self.rt.perceive(p);
        self.pending += 1;
        // Backpressure: too many unconsolidated perceptions ⇒ dream now, don't wait for the interval.
        if self.cfg.pressure_watermark > 0 && self.pending >= self.cfg.pressure_watermark {
            self.breathe_all(BreathCause::Pressure);
        }
    }

    fn breathe_all(&mut self, cause: BreathCause) -> BreathReport {
        let subjects: Vec<String> = self.subjects.iter().cloned().collect();
        self.breathe_some(&subjects, cause)
    }

    fn breathe_some(&mut self, subjects: &[String], cause: BreathCause) -> BreathReport {
        let refs: Vec<&str> = subjects.iter().map(String::as_str).collect();
        let report = self.rt.breathe(&refs, self.now());

        self.stats.breaths += 1;
        match cause {
            BreathCause::Basal => self.stats.breaths_basal += 1,
            BreathCause::Pressure => self.stats.breaths_pressure += 1,
            BreathCause::OnDemand => self.stats.breaths_ondemand += 1,
        }
        self.stats.distilled_subjects += report.distilled_subjects as u64;
        self.stats.perceptions_absorbed += report.perceptions_absorbed as u64;
        self.stats.faded += report.faded as u64;
        self.pending = 0;
        report
    }

    fn snapshot(&self) -> Stats {
        Stats {
            pending: self.pending,
            short_term_len: self.rt.short_term_len(),
            long_term_len: self.rt.long_term_len(),
            ..self.stats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_inference::MockProvider;

    fn cfg() -> AsyncConfig {
        AsyncConfig {
            breath_interval: Duration::from_secs(10),
            pressure_watermark: 0, // basal and on-demand tests don't want pressure interference
            ..Default::default()
        }
    }

    fn perception(subject: &str, e: Vec<f32>) -> Perception {
        // huge halflife: logical time barely advances in these tests, nothing should decay on its own.
        Perception::new(subject, e, 1.0, 1.0e9, 0.0)
    }

    #[tokio::test]
    async fn on_demand_breathe_consolidates() {
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        for _ in 0..20 {
            rt.perceive(perception("u:X", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        let report = rt.breathe(None).await.unwrap();
        assert_eq!(report.distilled_subjects, 1);
        assert_eq!(report.perceptions_absorbed, 20);

        let ctx = rt
            .evoke(EvokeRequest::new("u:X", 800))
            .await
            .unwrap()
            .expect("hay esencia");
        assert_eq!(ctx.represented, 20);
    }

    #[tokio::test(start_paused = true)]
    async fn basal_breathing_runs_without_being_asked() {
        // With clock paused, we control time: nobody calls breathe, but the interval fires the
        // dream on its own.
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        for _ in 0..5 {
            rt.perceive(perception("u:Y", vec![0.0, 1.0]))
                .await
                .unwrap();
        }
        // Before the interval: nothing consolidated.
        assert_eq!(rt.stats().await.unwrap().breaths, 0);

        // Advance the clock past the breathing interval.
        tokio::time::advance(Duration::from_secs(11)).await;
        // Yield so the actor processes the tick before we query.
        tokio::task::yield_now().await;

        let s = rt.stats().await.unwrap();
        assert!(s.breaths >= 1, "organism breathed on its own: {s:?}");
        assert_eq!(
            s.long_term_len, 1,
            "consolidated the subject without being asked"
        );
    }

    #[tokio::test]
    async fn backpressure_triggers_immediate_breath() {
        let pressured = AsyncConfig {
            pressure_watermark: 8,
            breath_interval: Duration::from_secs(3600), // far in the future: dream is NOT basal here
            ..cfg()
        };
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), pressured);
        for _ in 0..8 {
            rt.perceive(perception("u:Z", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        // 8th perception crosses the watermark ⇒ immediate dream, no waiting for interval or demand.
        let s = rt.stats().await.unwrap();
        assert_eq!(s.breaths, 1, "pressure triggered a dream: {s:?}");
        assert_eq!(s.long_term_len, 1);
        assert_eq!(s.pending, 0, "pressure was released");
    }

    #[tokio::test]
    async fn stops_cleanly_when_handle_dropped() {
        let (rt, handle) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        rt.perceive(perception("u:X", vec![1.0, 0.0]))
            .await
            .unwrap();
        drop(rt); // drop the only handle
                  // The actor must finish its task without panicking.
        handle.await.expect("actor finished cleanly");
    }

    #[tokio::test]
    async fn breath_causes_are_counted_separately() {
        let pressured = AsyncConfig {
            pressure_watermark: 4,
            breath_interval: Duration::from_secs(3600), // far in the future
            ..cfg()
        };
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), pressured);
        for _ in 0..4 {
            rt.perceive(perception("u:A", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        rt.breathe(None).await.unwrap(); // a demanda
        let s = rt.stats().await.unwrap();
        assert_eq!(s.breaths_pressure, 1, "{s:?}");
        assert_eq!(s.breaths_ondemand, 1, "{s:?}");
        assert_eq!(s.breaths_basal, 0);
        assert_eq!(s.breaths, 2);
    }

    #[tokio::test]
    async fn prometheus_render_is_well_formed() {
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        for _ in 0..3 {
            rt.perceive(perception("u:M", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        rt.breathe(None).await.unwrap();
        let text = rt.metrics_prometheus().await.unwrap();
        assert!(text.contains("# TYPE letheo_breaths_total counter"));
        assert!(text.contains("letheo_breaths_ondemand_total 1"));
        assert!(text.contains("# TYPE letheo_long_term_len gauge"));
        // Each metric has HELP + TYPE + value: 3 lines per metric, 10 metrics.
        assert_eq!(text.lines().count(), 30);
    }

    #[tokio::test]
    async fn evoke_on_unknown_subject_is_none() {
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        let ctx = rt.evoke(EvokeRequest::new("ghost", 800)).await.unwrap();
        assert!(ctx.is_none());
    }
}
