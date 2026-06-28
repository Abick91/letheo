//! # letheo-inference · Local-first inference kernel
//!
//! [`Provider`] abstraction decoupled from the runtime. Providers:
//! - `CandleProvider` (feature `candle`) — local `all-MiniLM-L6-v2`, 384-dim. **Product provider.**
//! - `MockProvider` (feature `testing`, or under `cfg(test)`) — deterministic embeddings, no model.
//!   It is a **test double**, NOT compiled into any product binary.

pub mod caching_provider;
pub mod provider;

// MockProvider exists only for tests: under the crate's own `cfg(test)`, or via the `testing` feature
// enabled by other crates' dev-dependencies. It never enters a product build.
#[cfg(any(test, feature = "testing"))]
pub mod mock_provider;
#[cfg(any(test, feature = "testing"))]
pub use mock_provider::MockProvider;

pub use caching_provider::{CacheStats, CachingProvider};
pub use provider::{Provider, EMBED_DIM};

#[cfg(feature = "candle")]
pub mod candle_provider;
#[cfg(feature = "candle")]
pub use candle_provider::CandleProvider;
