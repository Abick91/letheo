//! # letheo-core · The Letheo Cognitive Runtime
//!
//! Not a database: an organism that **perceives, dreams, evokes and fades**.
//!
//! - [`entropy`]  — The physics of forgetting (`weight(t) = salience·e^(−λ·Δt)·(1+reinforcement)`), lazy.
//! - [`perception`] — Volatile short-term memory (`PERCEIVE`).
//! - [`synthesis`] — The "dream": semantic compression via centroid + variance (`DISTILL`).
//! - [`archetype`] — Semantic long-term memory (layer-2), evolution anchor (`IMPRINT`).
//! - [`factstore`] — Episodic memory (layer-1): verbatim facts with forgetting, dedup and index.
//! - [`evoke`]     — Semantic resonance with a token budget (`EVOKE`).
//! - [`reflection`] — Generative memory: arc insights + predictive compression ("intelligence = compression").
//! - [`runtime`]   — The loop that "breathes" and orchestrates the layers.
//! - [`vector`]    — Vector operations (cosine, centroid), Flat search.

pub mod archetype;
pub mod entropy;
pub mod evoke;
pub mod factstore;
pub mod modes;
pub mod perception;
pub mod reflection;
pub mod runtime;
pub mod synthesis;
pub mod vector;

pub use archetype::{Archetype, ArchetypeStore, Resilience};
pub use entropy::{EntropyTrace, Tick};
pub use evoke::{
    approx_token_count, evoke, evoke_unified, ArcDetail, CompressedContext, EvokeRequest,
    UnifiedContext, DEFAULT_TOKENS_PER_VECTOR,
};
pub use factstore::{Fact, FactStore, RecalledFact, Remember};
pub use modes::{cluster_modes, Mode, ModeConfig, ModeSeed};
pub use perception::{Perception, PerceptionBuffer};
pub use reflection::{materialize, predictive_compression, reflect, Insight, PredictiveScore};
pub use runtime::{BreathReport, CognitiveRuntime, RuntimeConfig};
pub use synthesis::{distill, DistillConfig, IntentionVector};
