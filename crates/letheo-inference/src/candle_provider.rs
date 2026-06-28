//! `CandleProvider` — real local inference with `all-MiniLM-L6-v2` (BERT), 384-dim.
//!
//! Consistent with "local-first": the model lives **on disk** and runs in-process, with no network
//! or external service. The downloader (Python's huggingface_hub or any `git lfs`) runs once out of
//! band. This avoids coupling the runtime to a specific HTTP client.
//!
//! Only compiled with `--features candle`.

use crate::provider::Provider;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Model identifier (for documentation / download).
pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Environment variable pointing to the model directory on disk.
pub const MODEL_DIR_ENV: &str = "LETHEO_MODEL_DIR";

/// Local BERT-based embedding provider (Candle). 384 dimensions.
pub struct CandleProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    // BERT's forward shares tensors; we serialize access to be Sync-safe.
    lock: Mutex<()>,
}

impl CandleProvider {
    /// Loads the model from the directory pointed to by `LETHEO_MODEL_DIR`.
    ///
    /// The directory must contain `config.json`, `tokenizer.json` and `model.safetensors`
    /// (run `python sandbox/fetch_model.py` once to populate it).
    pub fn load() -> Result<Self> {
        let dir = std::env::var(MODEL_DIR_ENV).map_err(|_| {
            anyhow::anyhow!(
                "set {MODEL_DIR_ENV} pointing to the model directory \
                 (run `python sandbox/fetch_model.py` to download it)"
            )
        })?;
        Self::from_dir(dir)
    }

    /// Loads `all-MiniLM-L6-v2` from a local directory. CPU by default.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let device = Device::Cpu;

        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");
        for p in [&config_path, &tokenizer_path, &weights_path] {
            anyhow::ensure!(
                p.exists(),
                "{} missing from the model directory",
                p.display()
            );
        }

        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .context("parsing config.json")?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            lock: Mutex::new(()),
        })
    }

    /// Raw embedding (Result) — mean-pooling over tokens + L2 normalization.
    fn embed_inner(&self, text: &str) -> Result<Vec<f32>> {
        let _guard = self.lock.lock().unwrap();

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids = encoding.get_ids().to_vec();
        let n = ids.len();

        let token_ids = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::ones((1, n), DType::U8, &self.device)?;

        // forward → (1, seq_len, hidden=384)
        let out = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling over the token dimension → (384,)
        let (_b, seq, _h) = out.dims3()?;
        let pooled = (out.sum(1)? / seq as f64)?.squeeze(0)?;

        // L2 normalization.
        let norm = pooled.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let v: Vec<f32> = pooled.to_vec1()?;
        Ok(if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        })
    }
}

impl Provider for CandleProvider {
    fn dim(&self) -> usize {
        384
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // TRUTH 100% (debt #7): an inference failure with the model ALREADY loaded is exceptional and
        // **fails loudly** — a silent fake embedding is never returned (a zero vector would contaminate
        // centroids and resonances unnoticed). *Load* errors are surfaced earlier, via
        // `load()`/`from_dir()`.
        self.embed_inner(text).unwrap_or_else(|e| {
            panic!("CandleProvider: inference failure (model already loaded): {e:#}")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Requires the model on disk. Populate with `python sandbox/fetch_model.py` and export
    // LETHEO_MODEL_DIR. Ignored by default in CI without the model.
    #[test]
    #[ignore = "requires LETHEO_MODEL_DIR with all-MiniLM-L6-v2; run with --ignored"]
    fn loads_and_embeds_384() {
        let p = CandleProvider::load().expect("model load");
        let a = p.embed("running shoes at night");
        let b = p.embed("sneakers for nocturnal jogging");
        let c = p.embed("mortgage insurance bank loan");
        assert_eq!(a.len(), 384);

        let cos = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(i, j)| i * j).sum::<f32>();
        assert!(
            cos(&a, &b) > cos(&a, &c),
            "related phrases should resonate more than unrelated ones"
        );
    }
}
