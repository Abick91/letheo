<div align="center">

# Letheo

### Your AI agent doesn't have a memory problem. It has a **forgetting** problem.

**A Cognitive Runtime for agent memory** — an organism that perceives, dreams, evokes, and *fades*.
Memory at **constant cost**, whether the agent's history is 4,000 or 1,000,000 events.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/core-Rust-orange.svg)
![Python](https://img.shields.io/badge/SDK-Python-3776AB.svg)
![Local-first](https://img.shields.io/badge/local--first-no%20network-success.svg)
![Tests](https://img.shields.io/badge/cargo%20test-144%20passed-success.svg)
![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)

</div>

---

> **Letheo is not a database. It's a *Cognitive Runtime*** — it doesn't "store and query"; it
> **perceives, dreams, evokes, and fades**.

## Why not just RAG?

When an agent's history grows, today's options break at a fixed token budget:

- **Stuff the whole past into the prompt** → unbounded, **O(N)** cost per turn.
- **Re-summarize with an LLM every step** → still O(N), plus latency and drift.
- **RAG** → retrieves point facts but is **blind to time**: it ranks by *similarity*, not *recency*.
  It doesn't know that something *changed*, and its vector store grows forever.

Letheo distills behaviour into a **fixed-size** structure read at **constant cost**.

| | RAG | Letheo |
|---|---|---|
| Memory model | Store that **grows** ∞ | **Fixed-size** distilled essence |
| Cost per recall | O(N) in history | **O(1)** — flat at 4k or 1M events |
| Recall criterion | Cosine similarity | `relevance · weight(now)` (decay + salience + reinforcement) |
| Time | Invisible | A coefficient of entropy |
| Forgetting | None (or manual TTL) | **Native & physical** — nothing is immortal |
| Exact facts | Mixed into the corpus | Separate **verbatim layer**, same physics |

**Strategic forgetting is a feature, not a bug**: each memory's weight decays by physics (temporal
entropy) and only the pattern survives. Letheo is built to be the **memory of a fleet of
super-agents**: a single decay physics over **two layers** — episodic (exact facts, hippocampus) and
semantic (identity / trajectory, neocortex).

## The verbs (MQL — *Mnemonic Query Language*)

There is no `SELECT / INSERT / UPDATE / DELETE`. The vocabulary is biological:

| Verb | Role |
|------|------|
| `PERCEIVE` | Take in a raw stimulus into volatile short-term memory. It is born decaying. |
| `DISTILL`  | The "dream": collapse N perceptions into an *Intention Vector* + its **modes** (multi-modal compression). |
| `EVOKE`    | Recall by **semantic resonance** within a token budget; `RESONATING WITH` focuses on a trait. |
| `FADE`     | Strategic forgetting modulated by entropy; preserves the contribution already made to the archetype. |
| `IMPRINT`  | Consolidate / anchor an archetype against forgetting. |
| `RECALL`   | Layer-1: directed retrieval of **exact facts** (verbatim), read-only. |
| `REINFORCE`| Layer-1: spaced repetition — recall and reset a fact's decay. |

## Time as a coefficient of entropy

Time is not a timestamp; it's a passive operator on each memory's weight:

```
weight(t) = salience · e^(−λ · Δt) · (1 + reinforcement)        λ = ln2 / halflife
```

Δt is measured from the **last evocation/reinforcement** (recalling resets Δt → earned permanence).
Weight is evaluated **lazily**: only during `DISTILL`, `EVOKE`, or the semantic GC sweep — never per
clock tick. Reinforcement has **diminishing returns** and the half-life has a **floor**: nothing
becomes immortal no matter how often it's revisited.

## The two layers (Complementary Learning Systems)

A single physics (`EntropyTrace`) governs both representations of memory:

- **Layer-2 · semantic** (`archetype` + `modes`): the subject's identity and **trajectory**, decomposed
  into behavioural **modes** (not a blind average). Each mode has its own forgetting physics **and its
  own drift** (how far that behaviour has shifted since it was born). Compresses, O(1).
- **Layer-1 · episodic** (`factstore`): **verbatim** facts with an embedding, semantic dedup, and
  forgetting. Answers the exact, nominal thing that layer-2 would never store.

The **unified** `EVOKE` answers **character AND nominal** in a single evocation, splitting one token
budget across both layers.

## Usage (Python)

```python
from letheo_orchestration import Session

s = Session()

# Layer-2: perceive and "dream" → the essence (identity + trajectory, at fixed cost)
for _ in range(20):
    s.perceive("user:ada", act="reads sci-fi novels at night")
s.breathe()

# Layer-1: an exact, verbatim fact
s.remember("user:ada", "allergic to penicillin")

# A single evocation answers character (gist) AND nominal (facts)
ctx = s.evoke_unified("user:ada", "what does ada read?")
print(s.recall("user:ada", "allergies", k=1))     # [('allergic to penicillin', ...)]

# Generative memory: insights from the arc (transitions, revivals)
print(s.reflect("user:ada"))

# Similarity search across subjects (ANN at scale): route to the most relevant one
print(s.resonate("space opera fan", k=3))
```

…or the same engine as **MQL**:

```
PERCEIVE interaction FROM subject "user:ada" AS { act: reads, genre: scifi }
DISTILL  subject "user:ada" INTO intention_vector COMPRESSING BY semantic_variance
EVOKE    essence OF "user:ada" RESONATING WITH { nostalgia } WITHIN budget 800 tokens
RECALL   facts FROM subject "user:ada" RESONATING WITH { allergy } WHERE resonates > 0.6 WITHIN k 3
```

## Architecture

- **`crates/letheo-core`** (Rust): forgetting physics, perception, multi-modal synthesis, archetypes, factstore, unified evoke, reflection, runtime.
- **`crates/letheo-inference`** (Rust): `Provider` trait + `CandleProvider` (`all-MiniLM-L6-v2`, local).
- **`crates/letheo-mql`** + **`crates/letheo-exec`** (Rust): lexer + parser for the verbs → AST → executor.
- **`crates/letheo-index`** (Rust): ANN index (HNSW) + `Retriever` (Flat/HNSW with life-filtering).
- **`crates/letheo-{async,persist,calibration,cli}`** (Rust): Tokio actor runtime, persistence (JSON + embedded `redb` store), threshold calibration, MQL REPL.
- **`bindings/letheo-py`** (PyO3) + **`orchestration/`** (Python): high-level SDK (`Session`, prose, tiktoken).

```
crates/ + bindings/   →  ENGINE (Rust)           perceive · dream · evoke · forget
orchestration/        →  Python SDK (Session)    consumer layer over the binding
```

## Install

```bash
# 1) Engine (offline, hermetic) — no network, no model:
cargo test --workspace

# 2) Python binding (needs maturin + the local model in .models/):
maturin develop -m bindings/letheo-py/Cargo.toml --features candle
```

`CandleProvider` loads `all-MiniLM-L6-v2` **from disk** (local-first; it does not download at runtime).
Place it once and point `LETHEO_MODEL_DIR` at it:

```bash
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 .models/all-MiniLM-L6-v2
export LETHEO_MODEL_DIR="$PWD/.models/all-MiniLM-L6-v2"
```

Candle reads the config, tokenizer, and weights in **safetensors**. The Rust workspace
(`cargo test --workspace`) is **hermetic**: it doesn't need the model — only the Python binding does.

## Status

Engine (Rust), mature and tested offline: **`cargo test --workspace` → 144 passed, 0 failed, 2 ignored,
0 warnings**. Multi-modal archetype with per-mode trajectory, physical retrieval, unified episodic
two-layer memory, ANN index at scale, generative memory, transactional persistence — under the
**TRUTH 100%** invariant (zero mock/fake/hardcode on the product path).

---

<div align="center">

**v0.1.0 · MIT.** If memory that forgets by physics resonates with your agent's use case —
**star the repo** ⭐ and open an issue with your long-lived-agent scenario. I want to break it.

</div>
