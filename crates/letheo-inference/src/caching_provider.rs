//! Embedding cache (decorates any [`Provider`]).
//!
//! Embedding the same text twice is wasteful — with a real model (Candle/llama.cpp) it is the
//! most expensive operation in the pipeline. `CachingProvider` memoises `embed(text)` by exact
//! text, so repeated stimuli (habits: the same `act/object` a thousand times) are embedded once.
//!
//! Uses the full text as the key (not just a hash) → zero collisions. Interior mutability via
//! `Mutex` keeps it `Sync`, suitable for both the synchronous executor and the async actor.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::provider::Provider;

/// Wraps a `Provider` and caches its embeddings by text.
pub struct CachingProvider<P: Provider> {
    inner: P,
    cache: Mutex<HashMap<String, Vec<f32>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Cache statistics (observability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
}

impl CacheStats {
    /// Hit rate in `[0, 1]`. 1.0 if no queries have been made yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl<P: Provider> CachingProvider<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// The wrapped provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            entries: self.cache.lock().unwrap().len(),
        }
    }

    /// Clears the cache (does not reset accumulated counters).
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
}

impl<P: Provider> Provider for CachingProvider<P> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // Hot path: already cached?
        if let Some(v) = self.cache.lock().unwrap().get(text) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return v.clone();
        }
        // Miss: compute outside the lock (the real embedding can be slow) then store.
        self.misses.fetch_add(1, Ordering::Relaxed);
        let v = self.inner.embed(text);
        self.cache
            .lock()
            .unwrap()
            .insert(text.to_string(), v.clone());
        v
    }

    fn summarize(&self, fragments: &[&str]) -> String {
        self.inner.summarize(fragments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Provider that counts how many times it was actually asked to embed.
    struct CountingProvider {
        calls: AtomicUsize,
    }

    impl CountingProvider {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl Provider for CountingProvider {
        fn dim(&self) -> usize {
            2
        }
        fn embed(&self, text: &str) -> Vec<f32> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            // Trivial deterministic embedding: length and first byte.
            vec![text.len() as f32, text.bytes().next().unwrap_or(0) as f32]
        }
    }

    #[test]
    fn repeated_text_is_embedded_once() {
        let p = CachingProvider::new(CountingProvider::new());
        let a = p.embed("purchase shoes");
        let b = p.embed("purchase shoes");
        let c = p.embed("purchase shoes");
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(
            p.inner().calls(),
            1,
            "embedded only once despite 3 queries"
        );

        let s = p.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 2);
        assert_eq!(s.entries, 1);
    }

    #[test]
    fn distinct_texts_are_separate_entries() {
        let p = CachingProvider::new(CountingProvider::new());
        let x = p.embed("alpha");
        let y = p.embed("beta");
        assert_ne!(x, y);
        assert_eq!(p.inner().calls(), 2);
        assert_eq!(p.stats().entries, 2);
    }

    #[test]
    fn hit_rate_reflects_usage() {
        let p = CachingProvider::new(CountingProvider::new());
        for _ in 0..9 {
            p.embed("same");
        }
        p.embed("other");
        // 10 queries: 2 misses (same, other) + 8 hits.
        let s = p.stats();
        assert_eq!(s.hits, 8);
        assert_eq!(s.misses, 2);
        assert!((s.hit_rate() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn clear_empties_cache_but_keeps_counters() {
        let p = CachingProvider::new(CountingProvider::new());
        p.embed("x");
        p.clear();
        p.embed("x"); // recomputed after clearing
        assert_eq!(p.inner().calls(), 2);
        assert_eq!(p.stats().misses, 2);
        assert_eq!(p.stats().entries, 1);
    }

    #[test]
    fn dim_delegates_to_inner() {
        let p = CachingProvider::new(CountingProvider::new());
        assert_eq!(p.dim(), 2);
    }
}
