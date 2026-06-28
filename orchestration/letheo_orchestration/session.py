"""``Session`` — high-level ergonomic API over ``letheo.Runtime``.

Designed so an agent (or a human) talks to Letheo as if to a memory organ, not a database.

Example:

    from letheo_orchestration import Session

    with Session() as mem:
        # Declarative style (executable MQL):
        mem.run('''
            PERCEIVE interaction FROM subject "user:Xolotl"
                     AS { act: purchase, object: shoes, hue: nocturnal }
            DISTILL subject "user:Xolotl" INTO intention_vector COMPRESSING BY semantic_variance
            EVOKE essence OF "user:Xolotl" WITHIN budget 800 tokens
        ''')

        # Python style (sugar over PERCEIVE):
        mem.perceive("user:Xolotl", act="purchase", object="running_shoes")

        # Recall + prose ready for an LLM:
        prompt_block = mem.prompt("user:Xolotl", token_budget=800)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import letheo

from .prose import to_prose
from .tokens import count_tokens, count_tokens_method


@dataclass(frozen=True)
class EvokeResult:
    """An evoked memory: the raw ``CompressedContext`` + its prose + the honest token count."""

    context: Any   # letheo.CompressedContext (not annotated to avoid coupling to the PyO3 type)
    prose: str
    #: REAL tokens of the prose block injected into the LLM (not the runtime heuristic).
    prose_tokens: int = 0
    #: "tiktoken" (exact) or "heuristic" (estimate) — for honest reporting.
    token_method: str = "heuristic"

    def __str__(self) -> str:
        return self.prose

    def fits(self, budget: int) -> bool:
        """Does the real text fit within the token budget?"""
        return self.prose_tokens <= budget


class Session:
    """A cognitive session. A friendly wrapper over ``letheo.Runtime``.

    It carries an internal logical clock (``now``) advanced with ``tick(seconds)``. This avoids
    passing ``now=...`` on every call and keeps the spirit of *time as an entropy coefficient*:
    the clock runs on its own between interactions, it is not a per-event stamp.
    """

    DEFAULT_HALFLIFE = 7 * 24 * 3600.0   # one week

    def __init__(
        self,
        *,
        halflife_secs: float | None = None,
        persist_path: str | os.PathLike | None = None,
    ) -> None:
        self._rt = letheo.Runtime()
        self._now: float = 0.0
        self._halflife = halflife_secs or self.DEFAULT_HALFLIFE
        self._subjects: set[str] = set()
        self._persist_path = os.fspath(persist_path) if persist_path is not None else None
        # If a persistence path is set and memory already exists, rehydrate it on open.
        if self._persist_path:
            self.load(self._persist_path)

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str | os.PathLike | None = None) -> int:
        """Persists long-term memory (per-subject snapshot). Returns how many were saved.

        Without an argument it uses ``persist_path`` (if set in the constructor).
        """
        target = os.fspath(path) if path is not None else self._persist_path
        if not target:
            raise ValueError("save() needs a path (or set persist_path in the constructor)")
        return self._rt.save(target)

    def load(self, path: str | os.PathLike | None = None) -> int:
        """Rehydrates long-term memory from a directory of snapshots.

        Returns how many archetypes were loaded; the restored subjects become available for
        ``evoke``/``breathe`` without having been perceived in this session.
        """
        target = os.fspath(path) if path is not None else self._persist_path
        if not target:
            raise ValueError("load() needs a path (or set persist_path in the constructor)")
        n = self._rt.load(target)
        self._subjects.update(self._rt.subjects)
        return n

    # ── Pythonic context ──────────────────────────────────────────────────
    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If persistence is set and the session exits cleanly, save a snapshot.
        if self._persist_path and exc_type is None:
            self.save(self._persist_path)
        return None

    # ── Clock ─────────────────────────────────────────────────────────────
    @property
    def now(self) -> float:
        return self._now

    def tick(self, seconds: float) -> "Session":
        """Advances the logical clock. Chainable."""
        if seconds < 0:
            raise ValueError("time does not run backwards in Letheo")
        self._now += seconds
        return self

    # ── Biological verbs ──────────────────────────────────────────────────
    def perceive(
        self,
        subject: str,
        *,
        salience: float = 0.7,
        halflife_secs: float | None = None,
        **traits: Any,
    ) -> "Session":
        """``PERCEIVE``: assimilates a stimulus. The kwargs are the traits (act, object, hue, ...)."""
        if not traits:
            raise ValueError("perceive() needs at least one trait (kwargs)")
        text = " ".join(f"{k} {v}" for k, v in sorted(traits.items()))
        self._rt.perceive(
            subject,
            text,
            salience=salience,
            halflife_secs=halflife_secs or self._halflife,
            now=self._now,
        )
        self._subjects.add(subject)
        return self

    def perceive_vector(
        self,
        subject: str,
        embedding,
        *,
        text: str = "",
        salience: float = 0.7,
        halflife_secs: float | None = None,
    ) -> "Session":
        """``PERCEIVE`` with a precomputed embedding (oracle / Candle / sentence-transformers).

        ``text`` is the lexical label that distillation retains to name the content.
        """
        self._rt.perceive_with_embedding(
            subject,
            list(embedding),
            text=text,
            salience=salience,
            halflife_secs=halflife_secs or self._halflife,
            now=self._now,
        )
        self._subjects.add(subject)
        return self

    def breathe(self, subjects: Iterable[str] | None = None) -> Any:
        """``DISTILL`` + ``FADE``: a sleep cycle. By default, over all seen subjects."""
        targets = list(subjects) if subjects is not None else sorted(self._subjects)
        return self._rt.breathe(targets, now=self._now)

    def evoke(self, subject: str, *, token_budget: int = 800, model: str | None = None) -> EvokeResult:
        """``EVOKE``: recall + prose + REAL token count of the injected block.

        ``model`` selects the tokenizer (if ``tiktoken`` is installed); otherwise a conservative
        heuristic over the real text is used.
        """
        ctx = self._rt.evoke(subject, token_budget=token_budget, now=self._now)
        prose = to_prose(ctx)
        return EvokeResult(
            context=ctx,
            prose=prose,
            prose_tokens=count_tokens(prose, model=model),
            token_method=count_tokens_method(model),
        )

    # ── Ergonomics ────────────────────────────────────────────────────────
    def prompt(self, subject: str, *, token_budget: int = 800) -> str:
        """Shortcut: returns only the prose block, ready to inject into an LLM."""
        return self.evoke(subject, token_budget=token_budget).prose

    # ── Layer-1 (exact facts) and generative memory ───────────────────────
    def remember(
        self,
        subject: str,
        text: str,
        *,
        provenance: str = "agent",
        salience: float = 0.9,
        halflife_secs: float | None = None,
    ) -> "Session":
        """Layer-1: records a **verbatim** episodic fact (lossless), under the forgetting physics."""
        self._rt.remember(
            subject,
            text,
            provenance=provenance,
            salience=salience,
            halflife_secs=halflife_secs or (30.0 * 86_400.0),
            now=self._now,
        )
        self._subjects.add(subject)
        return self

    def recall(self, subject: str, query: str, *, k: int = 3) -> list[tuple[str, str, float]]:
        """Layer-1: retrieves the ``k`` most relevant exact facts (and reinforces them)."""
        return self._rt.recall(subject, query, k=k, now=self._now)

    def evoke_unified(
        self, subject: str, query: str, *, token_budget: int = 800, fact_budget: int = 200
    ) -> dict:
        """Unified ``EVOKE``: character (layer-2) **and** nominal (layer-1) in a single evocation."""
        return self._rt.evoke_unified(
            subject, query, token_budget=token_budget, fact_budget=fact_budget, now=self._now
        )

    def reflect(self, subject: str) -> list[dict]:
        """Higher-order insights about the subject's arc (transitions, revivals)."""
        return self._rt.reflect(subject)

    def dream_reflect(self, subject: str) -> int:
        """Reflective sleep: materializes insights as high-salience facts (layer-1)."""
        return self._rt.dream_reflect(subject, now=self._now)

    def resonate(self, query: str, *, k: int = 5) -> list[str]:
        """**Similarity** search: the ``k`` subjects whose essence resonates most with the query.
        Uses the ANN index (HNSW) at scale, exact Flat below the threshold. For routing to the most
        relevant subject (fleet case)."""
        return self._rt.resonate(query, k=k, now=self._now)

    def validate(self, mql_src: str) -> list[str]:
        """Semantically validates an MQL program without executing it. Empty list ⇒ valid."""
        return letheo.validate_mql(mql_src)

    def run(self, mql_src: str, *, validate: bool = True) -> list[dict]:
        """Executes an MQL program. Returns a list of dicts (one per statement).

        With ``validate=True`` (default) it checks the semantics before executing and raises
        ``ValueError`` if there are problems, instead of executing the program halfway.
        """
        if validate:
            problems = letheo.validate_mql(mql_src)
            if problems:
                raise ValueError("invalid MQL:\n" + "\n".join(problems))
        return self._rt.execute_mql(mql_src, now=self._now)

    # ── Inspection ────────────────────────────────────────────────────────
    @property
    def short_term_len(self) -> int:
        return self._rt.short_term_len

    @property
    def long_term_len(self) -> int:
        return self._rt.long_term_len

    @property
    def subjects(self) -> list[str]:
        return sorted(self._subjects)

    @property
    def cache_stats(self) -> dict:
        """Embedding cache statistics: ``{hits, misses, entries, hit_rate}``.

        Useful to see how much is saved: in flows with repeated habits, ``hit_rate`` tends to 1.
        """
        return self._rt.cache_stats()
