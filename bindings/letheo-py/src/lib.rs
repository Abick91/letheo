//! # letheo (PyO3) — the Cognitive Runtime for agent memory in Python.
//!
//! Two usage modes:
//!  1. **Direct biological API**: `rt.perceive(...)`, `rt.breathe(...)`, `rt.evoke(...)`.
//!  2. **Executable MQL**: `rt.execute_mql(src)` parses and executes a complete MQL program.
//!
//! ```python
//! import letheo
//! rt = letheo.Runtime()
//! rt.execute_mql('''
//!     PERCEIVE interaction FROM subject "user:Xolotl" AS { act: purchase, object: shoes }
//!     DISTILL  subject "user:Xolotl" INTO intention_vector COMPRESSING BY semantic_variance
//!     EVOKE    essence OF "user:Xolotl" WITHIN budget 800 tokens
//! ''')
//! ```

use letheo_core::{
    approx_token_count, CognitiveRuntime, EvokeRequest, Insight, Perception, RuntimeConfig,
};
use letheo_exec::{ExecError, ExecResult, Executor};
use letheo_index::Retriever;
use letheo_inference::{CachingProvider, CandleProvider, Provider};

// The product binding is Candle-only: no Mock build exists.
#[cfg(not(feature = "candle"))]
compile_error!("letheo-py requires the `candle` feature (real embeddings); no Mock variant exists.");
use letheo_mql::{parse, validate};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "CompressedContext")]
#[derive(Clone)]
struct PyCompressedContext {
    #[pyo3(get)]
    subject: String,
    #[pyo3(get)]
    represented: usize,
    #[pyo3(get)]
    vectors_returned: usize,
    #[pyo3(get)]
    anomalies_included: usize,
    #[pyo3(get)]
    arc_points: Vec<(f64, f32)>,
    #[pyo3(get)]
    core_label: String,
    #[pyo3(get)]
    arc_labels: Vec<String>,
    #[pyo3(get)]
    anomaly_labels: Vec<String>,
    #[pyo3(get)]
    domain_arcs: Vec<(String, Vec<f32>)>,
    #[pyo3(get)]
    arc_label_histograms: Vec<Vec<(String, usize)>>,
    #[pyo3(get)]
    token_estimate: usize,
    #[pyo3(get)]
    compression_ratio: f64,
    /// Mode the evocation focused on via `RESONATING WITH` (layer-2), or `None`.
    #[pyo3(get)]
    resonating_mode: Option<String>,
    /// Trajectory per live mode: `[(label, drift)]` — how much each behaviour has shifted from its origin.
    #[pyo3(get)]
    mode_drifts: Vec<(String, f32)>,
}

#[pymethods]
impl PyCompressedContext {
    fn __repr__(&self) -> String {
        format!(
            "CompressedContext(subject='{}', represented={}, arc_pts={}, token_estimate={}, ratio={:.1}:1)",
            self.subject,
            self.represented,
            self.arc_points.len(),
            self.token_estimate,
            self.compression_ratio
        )
    }
}

#[pyclass(name = "BreathReport")]
#[derive(Clone)]
struct PyBreathReport {
    #[pyo3(get)]
    distilled_subjects: usize,
    #[pyo3(get)]
    perceptions_absorbed: usize,
    #[pyo3(get)]
    faded: usize,
}

#[pymethods]
impl PyBreathReport {
    fn __repr__(&self) -> String {
        format!(
            "BreathReport(distilled_subjects={}, perceptions_absorbed={}, faded={})",
            self.distilled_subjects, self.perceptions_absorbed, self.faded
        )
    }
}

fn ctx_to_py(c: &letheo_core::CompressedContext) -> PyCompressedContext {
    PyCompressedContext {
        subject: c.subject.clone(),
        represented: c.represented,
        vectors_returned: c.vectors_returned,
        anomalies_included: c.anomalies_included,
        arc_points: c.arc_points.clone(),
        core_label: c.core_label.clone(),
        arc_labels: c.arc_labels.clone(),
        anomaly_labels: c.anomaly_labels.clone(),
        domain_arcs: c.domain_arcs.clone(),
        arc_label_histograms: c.arc_label_histograms.clone(),
        token_estimate: c.token_estimate,
        compression_ratio: c.compression_ratio(),
        resonating_mode: c.resonating_mode.clone(),
        mode_drifts: c.mode_drifts.clone(),
    }
}

// Provider for the live runtime: real Candle (all-MiniLM-L6-v2). The core crates are untouched;
// this is PyO3 bridge wiring. No Mock variant — Mock lives only in core tests.
type RuntimeProvider = CachingProvider<CandleProvider>;

fn make_provider() -> PyResult<RuntimeProvider> {
    // Real embeddings. Requires LETHEO_MODEL_DIR pointing to the model on disk.
    let p = CandleProvider::load().map_err(|e| {
        PyOSError::new_err(format!(
            "could not load Candle model (set LETHEO_MODEL_DIR to the all-MiniLM-L6-v2 dir): {e}"
        ))
    })?;
    Ok(CachingProvider::new(p))
}

/// The Cognitive Runtime. Internally holds an Executor with a real (Candle) provider that serves
/// both the direct API and MQL execution. The embedding cache embeds each text only once.
#[pyclass(name = "Runtime")]
struct PyRuntime {
    exec: Executor<RuntimeProvider>,
    // Similarity search at scale: exact Flat below the size threshold, HNSW above (caches the index).
    retriever: Retriever,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            exec: Executor::new(
                CognitiveRuntime::new(RuntimeConfig::default()),
                make_provider()?,
            ),
            retriever: Retriever::new(256),
        })
    }

    /// Direct `PERCEIVE`: assimilates a raw stimulus (text → embedding via local provider).
    #[pyo3(signature = (subject, text, salience=1.0, halflife_secs=64800.0, now=0.0))]
    fn perceive(
        &mut self,
        subject: &str,
        text: &str,
        salience: f64,
        halflife_secs: f64,
        now: f64,
    ) {
        let embedding = self.exec.provider().embed(text);
        // Store the raw text as a trait: it is the lexical label that distillation retains so that
        // prose can name the content (not just vectors).
        let perception =
            Perception::new(subject, embedding, salience, halflife_secs, now).with_trait("text", text);
        self.exec.runtime_mut().perceive(perception);
    }

    /// `PERCEIVE` with a **pre-computed** embedding (oracle / Candle / sentence-transformers).
    /// Bypasses the internal provider: the embedding comes from outside. `text` is stored as a
    /// lexical label so distillation can name the content.
    #[pyo3(signature = (subject, embedding, text="", salience=1.0, halflife_secs=64800.0, now=0.0))]
    fn perceive_with_embedding(
        &mut self,
        subject: &str,
        embedding: Vec<f32>,
        text: &str,
        salience: f64,
        halflife_secs: f64,
        now: f64,
    ) {
        let p = Perception::new(subject, embedding, salience, halflife_secs, now)
            .with_trait("text", text);
        self.exec.runtime_mut().perceive(p);
    }

    /// One "sleep" cycle: DISTILL → IMPRINT for the given subjects, then FADE noise.
    fn breathe(&mut self, subjects: Vec<String>, now: f64) -> PyBreathReport {
        let refs: Vec<&str> = subjects.iter().map(|s| s.as_str()).collect();
        let r = self.exec.runtime_mut().breathe(&refs, now);
        PyBreathReport {
            distilled_subjects: r.distilled_subjects,
            perceptions_absorbed: r.perceptions_absorbed,
            faded: r.faded,
        }
    }

    /// Direct `EVOKE`: resolves the essence of a subject within the token budget.
    #[pyo3(signature = (subject, token_budget=800, now=0.0))]
    fn evoke(&self, subject: &str, token_budget: usize, now: f64) -> PyResult<PyCompressedContext> {
        let req = EvokeRequest::new(subject, token_budget);
        match self.exec.runtime().evoke(&req, now) {
            Some(c) => Ok(ctx_to_py(&c)),
            None => Err(PyValueError::new_err(format!("no live essence for '{subject}'"))),
        }
    }

    /// **Layer-1** (`remember`): stores a verbatim episodic fact (text → embedding via provider),
    /// under forgetting physics. Semantic dedup per subject. High salience makes it durable.
    #[pyo3(signature = (subject, text, provenance="agent", salience=0.9, halflife_secs=2592000.0, now=0.0))]
    fn remember(
        &mut self,
        subject: &str,
        text: &str,
        provenance: &str,
        salience: f64,
        halflife_secs: f64,
        now: f64,
    ) {
        let embedding = self.exec.provider().embed(text);
        self.exec
            .runtime_mut()
            .remember(subject, text, embedding, provenance, salience, halflife_secs, now);
    }

    /// **Layer-1** (`recall`): retrieves the `k` most relevant exact facts for a subject (by
    /// physics) and **reinforces them** (spaced repetition). Returns `[(text, provenance, score)]`, verbatim.
    #[pyo3(signature = (subject, query, k=3, now=0.0))]
    fn recall(&mut self, subject: &str, query: &str, k: usize, now: f64) -> Vec<(String, String, f64)> {
        let q = self.exec.provider().embed(query);
        self.exec
            .runtime_mut()
            .recall(subject, &q, k, now)
            .into_iter()
            .map(|f| (f.text, f.provenance, f.score))
            .collect()
    }

    /// **Unified EVOKE**: a single evocation that answers character (layer-2) **and** nominal
    /// (layer-1) under ONE budget. Returns `{gist: CompressedContext|None, facts: [(t,prov,score)],
    /// fact_tokens, total_tokens}`. Fact cost is measured with the core estimator; the orchestration
    /// layer can inject tiktoken for exact counting.
    #[pyo3(signature = (subject, query, token_budget=800, fact_budget=200, now=0.0))]
    fn evoke_unified<'py>(
        &self,
        py: Python<'py>,
        subject: &str,
        query: &str,
        token_budget: usize,
        fact_budget: usize,
        now: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let q = self.exec.provider().embed(query);
        let req = EvokeRequest::new(subject, token_budget);
        let u = self.exec.runtime().evoke_unified(&req, &q, fact_budget, now, approx_token_count);
        let d = PyDict::new(py);
        d.set_item("gist", u.gist.as_ref().map(ctx_to_py))?;
        let facts: Vec<(String, String, f64)> =
            u.facts.iter().map(|f| (f.text.clone(), f.provenance.clone(), f.score)).collect();
        d.set_item("facts", facts)?;
        d.set_item("fact_tokens", u.fact_tokens)?;
        d.set_item("total_tokens", u.total_tokens)?;
        Ok(d)
    }

    /// **Reflection** (generative layer): higher-order insights about the subject's arc —
    /// transitions and revivals absent from any individual event. Returns a list of dicts.
    fn reflect<'py>(&self, py: Python<'py>, subject: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for ins in self.exec.runtime().reflect(subject) {
            let d = PyDict::new(py);
            match ins {
                Insight::Transition { from, to, support } => {
                    d.set_item("kind", "transition")?;
                    d.set_item("from", from)?;
                    d.set_item("to", to)?;
                    d.set_item("support", support)?;
                }
                Insight::Revival { domain } => {
                    d.set_item("kind", "revival")?;
                    d.set_item("domain", domain)?;
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// **Reflective sleep**: reflects and **materialises** insights as high-salience facts in
    /// layer-1 (retrievable via `recall`). Returns how many were stored.
    #[pyo3(signature = (subject, now=0.0))]
    fn dream_reflect(&mut self, subject: &str, now: f64) -> usize {
        self.exec.runtime_mut().dream_reflect(subject, now)
    }

    /// **Similarity search** (not by id): the `k` subjects whose essence resonates most with
    /// `query`. Uses the ANN index (HNSW) above the size threshold, exact Flat below, **filtering
    /// by life**. For routing a task to the most relevant agent/subject (Paideia fleet use-case).
    #[pyo3(signature = (query, k=5, now=0.0))]
    fn resonate(&mut self, query: &str, k: usize, now: f64) -> Vec<String> {
        let q = self.exec.provider().embed(query);
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        self.retriever
            .resonate_subjects(self.exec.runtime().long_term(), &q, k, now, theta)
    }

    /// Executes a complete MQL program. Returns a list of dicts, one per statement, of the form
    /// `{"kind": "...", ...fields}`. Per-statement errors go in `{"kind": "error", "message": "..."}`.
    #[pyo3(signature = (src, now=0.0))]
    fn execute_mql<'py>(
        &mut self,
        py: Python<'py>,
        src: &str,
        now: f64,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let stmts = parse(src).map_err(|e| PyValueError::new_err(e.message))?;
        let mut out = Vec::with_capacity(stmts.len());
        for stmt in &stmts {
            let d = PyDict::new(py);
            match self.exec.execute(stmt, now) {
                Ok(ExecResult::Perceived { subject }) => {
                    d.set_item("kind", "perceived")?;
                    d.set_item("subject", subject)?;
                }
                Ok(ExecResult::Dreamed(r)) => {
                    d.set_item("kind", "dreamed")?;
                    d.set_item("distilled_subjects", r.distilled_subjects)?;
                    d.set_item("perceptions_absorbed", r.perceptions_absorbed)?;
                    d.set_item("faded", r.faded)?;
                }
                Ok(ExecResult::Evoked(c)) => {
                    d.set_item("kind", "evoked")?;
                    d.set_item("context", ctx_to_py(&c).into_pyobject(py)?)?;
                }
                Ok(ExecResult::Faded { swept }) => {
                    d.set_item("kind", "faded")?;
                    d.set_item("swept", swept)?;
                }
                Ok(ExecResult::Imprinted { archetype, note }) => {
                    d.set_item("kind", "imprinted")?;
                    d.set_item("archetype", archetype)?;
                    d.set_item("note", note)?;
                }
                Ok(ExecResult::Recalled(facts)) => {
                    d.set_item("kind", "recalled")?;
                    let items: Vec<(String, String, f64)> =
                        facts.iter().map(|f| (f.text.clone(), f.provenance.clone(), f.score)).collect();
                    d.set_item("facts", items)?;
                }
                Ok(ExecResult::Reinforced { count }) => {
                    d.set_item("kind", "reinforced")?;
                    d.set_item("count", count)?;
                }
                Err(e) => {
                    d.set_item("kind", "error")?;
                    d.set_item("message", e.to_string())?;
                    d.set_item("variant", match e {
                        ExecError::NoSuchSubject(_) => "no_such_subject",
                        ExecError::MissingBudget => "missing_budget",
                    })?;
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// Persists **both layers** to `dir`: layer-2 (one JSON snapshot per archetype) and layer-1
    /// (`facts.json`). Returns how many archetypes were saved. Memory survives restarts.
    fn save(&self, dir: &str) -> PyResult<usize> {
        let n = letheo_persist::save_store(dir, self.exec.runtime().long_term())
            .map_err(|e| PyOSError::new_err(format!("could not save to '{dir}': {e}")))?;
        letheo_persist::save_facts(dir, self.exec.runtime().facts())
            .map_err(|e| PyOSError::new_err(format!("could not save facts to '{dir}': {e}")))?;
        Ok(n)
    }

    /// Rehydrates **both layers** from `dir` (archetypes + facts). Replaces the current memory.
    /// Returns how many archetypes were loaded. A non-existent directory loads 0 (first boot).
    fn load(&mut self, dir: &str) -> PyResult<usize> {
        let store = letheo_persist::load_store(dir)
            .map_err(|e| PyOSError::new_err(format!("could not load from '{dir}': {e}")))?;
        let facts = letheo_persist::load_facts(dir)
            .map_err(|e| PyOSError::new_err(format!("could not load facts from '{dir}': {e}")))?;
        let n = store.len();
        *self.exec.runtime_mut().long_term_mut() = store;
        *self.exec.runtime_mut().facts_mut() = facts;
        Ok(n)
    }

    /// Subjects with a consolidated essence in long-term memory (e.g. after `load`).
    #[getter]
    fn subjects(&self) -> Vec<String> {
        self.exec.runtime().long_term().iter().map(|a| a.subject.clone()).collect()
    }

    #[getter]
    fn short_term_len(&self) -> usize {
        self.exec.runtime().short_term_len()
    }

    #[getter]
    fn long_term_len(&self) -> usize {
        self.exec.runtime().long_term_len()
    }

    /// Embedding cache statistics: `{hits, misses, entries, hit_rate}`.
    fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.exec.provider().stats();
        let d = PyDict::new(py);
        d.set_item("hits", s.hits)?;
        d.set_item("misses", s.misses)?;
        d.set_item("entries", s.entries)?;
        d.set_item("hit_rate", s.hit_rate())?;
        Ok(d)
    }
}

/// `Embedder` — real embeddings with Candle (`all-MiniLM-L6-v2`, 384-dim), local and offline.
///
/// Only exists if the binding was compiled with `--features candle`. Computes vectors in Python
/// for injection via `Runtime.perceive_with_embedding` / `Session.perceive_vector` — the same
/// plug the test harness oracle uses, but with real semantics. Loads the model from
/// `LETHEO_MODEL_DIR` (populate with `python sandbox/fetch_model.py`).
#[cfg(feature = "candle")]
#[pyclass(name = "Embedder")]
struct PyEmbedder {
    inner: letheo_inference::CandleProvider,
}

#[cfg(feature = "candle")]
#[pymethods]
impl PyEmbedder {
    /// Loads the model from the directory at `LETHEO_MODEL_DIR`.
    #[new]
    fn new() -> PyResult<Self> {
        let inner = letheo_inference::CandleProvider::load()
            .map_err(|e| PyOSError::new_err(format!("could not load Candle model: {e}")))?;
        Ok(Self { inner })
    }

    /// Loads the model from an explicit directory (config.json, tokenizer.json, model.safetensors).
    #[staticmethod]
    fn from_dir(dir: &str) -> PyResult<Self> {
        let inner = letheo_inference::CandleProvider::from_dir(dir)
            .map_err(|e| PyOSError::new_err(format!("could not load Candle model: {e}")))?;
        Ok(Self { inner })
    }

    /// Dimension of the embeddings (384).
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Embeds a text → L2-normalised vector of 384 dimensions.
    fn embed(&self, text: &str) -> Vec<f32> {
        self.inner.embed(text)
    }
}

/// Parses an MQL program and returns the number of statements (without executing).
#[pyfunction]
fn parse_mql(src: &str) -> PyResult<usize> {
    parse(src).map(|s| s.len()).map_err(|e| PyValueError::new_err(e.message))
}

/// Semantically validates an MQL program. Returns the list of problems (empty ⇒ valid).
/// Syntax errors are raised as `ValueError`.
#[pyfunction]
fn validate_mql(src: &str) -> PyResult<Vec<String>> {
    let stmts = parse(src).map_err(|e| PyValueError::new_err(e.message))?;
    Ok(validate(&stmts).iter().map(|p| p.to_string()).collect())
}

#[pymodule]
fn letheo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyCompressedContext>()?;
    m.add_class::<PyBreathReport>()?;
    #[cfg(feature = "candle")]
    m.add_class::<PyEmbedder>()?;
    m.add_function(wrap_pyfunction!(parse_mql, m)?)?;
    m.add_function(wrap_pyfunction!(validate_mql, m)?)?;
    Ok(())
}
