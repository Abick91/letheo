# Security Policy

## Reporting a vulnerability
Please **do not** open a public issue for security problems. Email **security@totem-systems.com**
with the details and (if possible) a minimal reproduction. You'll get an acknowledgement within a
few business days, and we'll coordinate a fix + disclosure timeline with you.

## Scope
Letheo is a local-first memory engine (Rust core + optional Python SDK). The most relevant areas:
- **Persistence** — JSON snapshots / embedded `redb` store of archetypes & facts on local disk. Treat
  the data directory as sensitive (it can hold verbatim episodic facts).
- **Inputs** — embeddings/text fed to `PERCEIVE`/`DISTILL`/`EVOKE`. Report any path that lets crafted
  input crash the runtime or read outside its data directory.

## Out of scope
- The downloaded embedding model (`.models/`, all-MiniLM-L6-v2) is a third-party artifact.
- Resource exhaustion from intentionally pathological local input (it's a local-first engine).

## Good to know
The engine is offline and deterministic by design; there are no outbound network calls in the core
(`cargo test --workspace` is hermetic). Secrets are never committed — `.env`, `.models/` and `data/`
are gitignored.
