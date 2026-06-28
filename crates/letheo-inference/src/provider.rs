//! Inference provider abstraction — the decoupled "thinking engine".
//!
//! Local-first: inference happens in-process, with no network. The trait allows swapping the engine
//! (deterministic Mock, local Candle) without altering the runtime logic.

/// Embedding dimension (all-MiniLM-L6-v2).
pub const EMBED_DIM: usize = 384;

/// An inference engine able to produce semantic embeddings and summaries.
pub trait Provider {
    /// Dimension of the embeddings it produces.
    fn dim(&self) -> usize;

    /// Converts raw text into a normalizable dense embedding.
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Summarizes/expresses a set of traits as prose (for the context the LLM consumes).
    /// The default concatenates the fragments; a semantic provider (llama.cpp) can override it.
    fn summarize(&self, fragments: &[&str]) -> String {
        fragments.join(" · ")
    }
}
