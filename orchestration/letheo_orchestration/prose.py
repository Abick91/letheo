"""Narrative prose generation from a ``CompressedContext``.

The runtime delivers a **structured** context (core, anomalies, arc milestones). For a consuming LLM
to use it as memory, we turn it into an ultra-compressed block of **narrative prose**: dense natural
language, not lists of events.

Local-first: this module does NOT call an LLM to summarize; it *derives* the prose from the arc data.
Integration with an LLM (Claude/GPT) takes this block and injects it into the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ArcReading:
    """Interpreted reading of the subject's evolutionary arc."""

    direction: str    # "rising" | "falling" | "stable" | "volatile"
    drift_total: float
    drift_peak: float
    span_seconds: float
    inflexions: int    # number of sign changes in the drift derivative


def _ascii_sparkline(values: Sequence[float]) -> str:
    """Mini 8-level sparkline from a sequence of numbers."""
    if not values:
        return ""
    levels = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return levels[len(levels) // 2] * len(values)
    return "".join(levels[min(len(levels) - 1, int((v - lo) / (hi - lo) * (len(levels) - 1)))] for v in values)


# Noise floor: the ONLY embedder-dependent constant. Below a cosine shift of this order, the movement
# is below the model's resolution and reads as stable. Calibrated for all-MiniLM-L6-v2, which
# compresses related domains and produces peaks ~20x smaller than an orthogonal embedder (the SAME
# reversal: 0.375 with an oracle, 0.020 with MiniLM). The rest of the classification is scale-free
# (relative shape, not absolute magnitude).
ARC_NOISE_FLOOR = 0.015


def read_arc(arc_points: Sequence[tuple[float, float]]) -> ArcReading:
    """Interprets the arc as a qualitative reading of the trajectory.

    The *returned* (round-trip) detection is **scale-free**: it depends on SHAPE — the excursion peak
    dominates the net change and there was exactly one return — not on absolute magnitude. So the same
    arc reads the same with an orthogonal embedder (peaks ~0.4) or with one that compresses the domains
    (MiniLM, peaks ~0.02). The only absolute magnitude is `ARC_NOISE_FLOOR`.
    """
    if not arc_points:
        return ArcReading("stable", 0.0, 0.0, 0.0, 0)

    times = [t for (t, _) in arc_points]
    drifts = [d for (_, d) in arc_points]
    drift_total = drifts[-1] - drifts[0]
    drift_peak = max((abs(d) for d in drifts), default=0.0)
    span = times[-1] - times[0]

    # Inflexion count (sign changes in the discrete derivative).
    inflexions = 0
    deltas = [drifts[i + 1] - drifts[i] for i in range(len(drifts) - 1)]
    for i in range(len(deltas) - 1):
        if deltas[i] * deltas[i + 1] < 0:
            inflexions += 1

    if drift_peak < ARC_NOISE_FLOOR:
        # Below the embedder's resolution: no real excursion or direction.
        direction = "stable"
    # Several direction swings = volatile; ONE return with a peak that dominates the net change =
    # reversal (returned). The criterion is relative (peak ≫ net), not an absolute threshold: a
    # round-trip is NOT "stable" even if drift_total≈0.
    elif inflexions >= 2:
        direction = "volatile"
    elif inflexions == 1 and drift_peak >= 2.0 * abs(drift_total):
        direction = "returned"
    elif drift_total > 0.05:
        direction = "rising"
    elif drift_total < -0.05:
        direction = "falling"
    else:
        direction = "stable"

    return ArcReading(direction, drift_total, drift_peak, span, inflexions)


_DIRECTION_NARRATIVE = {
    "rising": "their identity has drifted steadily toward a new direction",
    "falling": "their identity has returned toward the initial pattern after exploring other territories",
    "returned": "their identity explored a different direction and later returned toward its initial pattern",
    "stable": "their identity has stayed consistent with its baseline",
    "volatile": "their identity has oscillated between opposing directions over the period",
}


def _domain_trend(series: list[float]) -> str | None:
    """Classifies the prevalence trajectory of ONE behaviour across the phases.

    `series` = the domain's fraction of activity at each milestone (0..1). Returns a narrative phrase
    or `None` if it was never relevant or there is no clear pattern (to avoid adding noise). Closes the
    per-domain reversal gap: the global arc does not distinguish which concrete behaviour returned.
    """
    if not series or len(series) < 2:
        return None
    peak = max(series)
    # Peak floor: below this is background noise (near-absent domains ~0.03), not a behaviour.
    if peak < 0.07:
        return None
    # SCALE-FREE: we normalize by the peak itself and classify by SHAPE, not magnitude. Essential
    # because a domain fragments into several templates → each with ~1/k of the real prevalence, but
    # the SAME shape. (Same principle as `read_arc`.)
    norm = [x / peak for x in series]
    n = len(norm)
    start, end = norm[0], norm[-1]
    interior_min = min(norm[1:-1]) if n > 2 else min(start, end)
    peak_idx = norm.index(1.0)
    if start >= 0.6 and interior_min <= 0.45 and end >= 0.6:
        return "faded for a season and then came back"
    if end < 0.25:
        if peak_idx == 0:
            return "has declined from its previous level"
        return "had a period of strong interest and then disappeared entirely"
    if end >= 0.85 and start <= 0.55:
        return "has been growing steadily"
    # Present without a marked pattern: we narrate it anyway (coverage) — a question about this domain
    # needs to see it even if it did not rise/fall. The peak floor already filtered the noise.
    return "has stayed a present interest"


# Minimal stopwords (EN + review-domain noise) so common terms are informative.
_STOP = {
    "this", "that", "with", "from", "have", "your", "just", "they", "them", "their", "what", "when",
    "which", "would", "could", "about", "there", "here", "into", "than", "then", "some", "more", "most",
    "very", "really", "much", "many", "like", "good", "great", "best", "well", "also", "even", "still",
    "movie", "movies", "film", "films", "review", "reviewed", "stars", "star", "watch", "watched",
    "story", "really", "dont", "didnt", "isnt", "thing", "things", "make", "made", "show", "shows",
    "text", "phrase",  # trait-key prefixes, not content (see _clean)
}


def _common_terms(histograms, k: int = 4) -> list[str]:
    """Most recurrent terms (frequency, weighted by count) among the milestone labels.

    Improvement E: instead of ONE representative title (uninformative in item-centric data), it derives
    the content words that repeat across the arc. It is presentation (prose layer); the engine only
    provides the histograms (additive field). Honest: it is frequency + stopwords, not full TF-IDF —
    enough to name the pattern without pretending otherwise.
    """
    import re
    from collections import Counter
    c: Counter = Counter()
    for hist in histograms or []:
        for label, count in hist:
            for tok in re.findall(r"[a-zA-Z]+", str(label).lower()):
                if len(tok) >= 4 and tok not in _STOP:
                    c[tok] += int(count)
    return [w for w, _ in c.most_common(k)]


def _is_faded_peak(series: list[float]) -> bool:
    """Did this behaviour have a real peak and already fade? (it mattered in the past, near-zero today).

    It is the `Q_past_peak` signal: what an agent must NOT forget *was* important. Scale-free
    (normalizes by the peak), same principle as `_domain_trend`.
    """
    if not series or len(series) < 2:
        return False
    peak = max(series)
    if peak < 0.07:  # never a real behaviour (background noise)
        return False
    return (series[-1] / peak) < 0.25  # ended far below its own peak


def to_prose(ctx, *, span_label: str = "the observed period") -> str:
    """Turns a ``CompressedContext`` into a prose block for an LLM.

    The block is meant to be injected as-is into a prompt: header, arc narrative, confidence metrics
    (compression, events represented) and a closing marker.

    Args:
        ctx: a ``letheo.CompressedContext`` instance.
        span_label: human text for the period (e.g. "the past year").
    """
    reading = read_arc(ctx.arc_points)
    spark = _ascii_sparkline([d for (_, d) in ctx.arc_points])

    direction_phrase = _DIRECTION_NARRATIVE.get(reading.direction, "their behaviour has evolved")

    lines = [
        f"≈ DISTILLED MEMORY · subject «{ctx.subject}» · {span_label}",
        "",
        f"Over {span_label}, {direction_phrase} (cumulative Δ = {reading.drift_total:+.2f}, "
        f"peak change = {reading.drift_peak:.2f}).",
    ]

    # Lexical content: what occupies them NOW (current core label). Without this, the prose narrates the
    # drift but does not name the topic — an LLM could not say "what they're up to".
    core_label = _clean(getattr(ctx, "core_label", ""))
    if core_label:
        lines.append(f"Right now their behaviour gravitates around: {core_label}.")

    # Improvement E (additive): recurrent terms across the arc — they name the PATTERN, not a single title.
    # Uses the new `arc_label_histograms` field if the binding exposes it; otherwise adds nothing.
    terms = _common_terms(getattr(ctx, "arc_label_histograms", None))
    if len(terms) >= 2:
        lines.append("Recurrent terms in their arc: " + ", ".join(terms) + ".")

    # Past peaks already faded: behaviours that MATTERED and went quiet. A dedicated section, placed
    # EARLY in the block to survive budget trimming — it is the content that answers Q_past_peak
    # ("what did they care about before and no longer?"), the engine's most robust win.
    faded = []
    faded_names = set()
    for label, series in (getattr(ctx, "domain_arcs", []) or []):
        name = _clean(label)
        if name and _is_faded_peak([float(x) for x in series]):
            faded.append(name)
            faded_names.add(name)
    if faded:
        lines.append("Past peaks already faded (they mattered and barely appear today — relevant if "
                     "asked about the past): " + ", ".join(f"«{n}»" for n in faded[:5]) + ".")

    # Per-behaviour evolution: the dimension the global arc does not capture — answers "did X come back?"
    # for a concrete behaviour. We exclude those already listed as past-peaks to avoid duplication.
    trend_lines = []
    for label, series in (getattr(ctx, "domain_arcs", []) or []):
        name = _clean(label)
        if not name or name in faded_names:
            continue
        phrase = _domain_trend([float(x) for x in series])
        if phrase:
            trend_lines.append(f"  · «{name}»: {phrase}")
    if trend_lines:
        lines.append("Per-behaviour evolution:")
        lines.extend(trend_lines)

    # Named trajectory: the sequence of dominant topics across the arc (without consecutive repeats),
    # e.g. "trail running → yoga → trail running" (captures reversals).
    arc_labels = [_clean(x) for x in getattr(ctx, "arc_labels", []) if _clean(x)]
    path = _dedup_consecutive(arc_labels)
    if len(path) >= 2:
        lines.append("Thematic trajectory: " + " → ".join(path) + ".")

    if ctx.arc_points:
        lines.append(f"Temporal signature of the arc: {spark}  ({len(ctx.arc_points)} milestones)")
        if reading.direction == "volatile":
            lines.append(
                f"{reading.inflexions} direction inflexions are observed — the subject changed "
                "direction and changed again; a careful agent should probe their current state "
                "before assuming continuity."
            )

    if ctx.anomalies_included:
        anom = [_clean(x) for x in getattr(ctx, "anomaly_labels", []) if _clean(x)]
        named = f" (e.g.: {'; '.join(anom[:3])})" if anom else ""
        lines.append(
            f"{ctx.anomalies_included} atypical signals persist that do not fit the behaviour "
            f"core{named}; treat them as live hypotheses, not as disposable anecdotes."
        )

    lines.extend([
        "",
        f"This memory represents the distilled trace of {ctx.represented:,} original "
        f"interactions, condensed to {ctx.vectors_returned} dense vectors "
        f"({_fmt_ratio(ctx.compression_ratio)}).",
        "",
        "≈ END MEMORY",
    ])
    return "\n".join(lines)
    # Honesty note: the REAL token count of the block is exposed by Session.evoke().prose_tokens
    # (real tokenizer / declared heuristic). We do not assert a "≤ N" here that text with lexical
    # labels may not satisfy; the budget is managed where the text exists.


def _clean(label: str) -> str:
    """Normalizes a lexical label for the prose: strips underscores and the trait prefix.

    The block stores the trait as ``"text <phrase>"`` or ``"phrase phrase_with_underscores"``; here we
    make it readable. It is presentation, not semantics.
    """
    if not label:
        return ""
    s = label.replace("_", " ").strip()
    # Strips a redundant trait-key prefix ("text ", "phrase ").
    for pref in ("text ", "phrase ", "act "):
        if s.startswith(pref):
            s = s[len(pref):]
    return s.strip()


def _dedup_consecutive(items: list[str]) -> list[str]:
    """Collapses consecutive repetitions: [a,a,b,a] → [a,b,a] (preserves reversals)."""
    out: list[str] = []
    for it in items:
        if not out or out[-1] != it:
            out.append(it)
    return out


def _fmt_ratio(r: float) -> str:
    if r >= 1000:
        return f"compression {r/1000:.1f}k:1"
    return f"compression {r:.1f}:1"
