# Contributing to Letheo

Letheo is a **Cognitive Runtime** — a memory engine that perceives, distills, evokes and *forgets*.
Contributions are welcome. A few invariants keep it trustworthy:

## The non-negotiable: TRUTH 100%
**No mocks, fakes, hardcodes, proxies or placeholders in the product path.** If a number can't be
measured, it's declared as an approximation (e.g. token counts), never invented. Test doubles live
only in tests, never beside real code. A PR that fakes a result will be rejected even if green.

## Layering (don't cross the boundaries)
```
crates/ + bindings/   →  ENGINE (Rust)           perceives · dreams · evokes · forgets
orchestration/        →  Python SDK (Session)    binding consumer layer
```
- New physics/algorithms → `crates/letheo-core` (and friends). Keep the core pure and offline.
- `letheo-py` is excluded from the workspace on purpose, so `cargo test --workspace` stays hermetic
  (no network, no model download). Don't add network/model deps to the workspace test path.

## Before you open a PR
1. `cargo test --workspace` must stay **green** (currently 115 passed, 0 failed, 1 ignored). Add tests
   for new behaviour — physics changes need a test that pins the property (decay, resonance, dedup…).
2. `cargo fmt` + `cargo clippy` clean.
3. Document design decisions and tradeoffs in your PR description.
4. Keep the engine **local-first**: the embedding model is loaded from `LETHEO_MODEL_DIR`
   (all-MiniLM-L6-v2 via Candle), never downloaded inside tests.

## Scope
The North Star is the memory engine. Benchmarks/duels/demos were deliberately pruned — proposals
to re-add comparison scaffolding should justify why.

By contributing you agree your contributions are licensed under the repository's MIT License.
