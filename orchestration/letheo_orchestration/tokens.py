"""Honest token counting of the memory block injected into the LLM.

Context: the Rust runtime exposes ``CompressedContext.token_estimate``, an **allocation estimate**
(``num_vectors × tokens_per_vector``, fed back from tiktoken) used *inside* ``evoke`` to decide how
many vectors fit. It is **not** the token count of the text finally sent to the model. To manage the
budget honestly you must count the **real text** of the prose with the consumer model's tokenizer.

Pluggable strategy:
- If ``tiktoken`` is installed → **exact** count with the model's encoding (OpenAI/DeepSeek).
- Otherwise → a heuristic calibrated **on the real text** (conservative, does not under-estimate the
  budget). It is an estimate, declared as such via ``count_tokens_method()``; the path to exact is
  ``pip install tiktoken``.
"""
from __future__ import annotations

from functools import lru_cache

# Default encoding of modern GPT-4o/4o-mini models. DeepSeek is reasonably close.
_DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=8)
def _try_tiktoken(model: str | None):
    """Returns a tiktoken encoder or ``None`` if not available offline."""
    try:
        import tiktoken
    except Exception:
        return None
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception:
        return None


def count_tokens_method(model: str | None = None) -> str:
    """``"tiktoken"`` (exact) or ``"heuristic"`` (estimate). For honest reporting."""
    return "tiktoken" if _try_tiktoken(model) is not None else "heuristic"


def _heuristic(text: str) -> int:
    """Conservative estimate on the real text.

    Two OpenAI rules of thumb: ~4 chars/token and ~0.75 words/token (≈ words×1.33). We take the
    **maximum** of both so as not to under-estimate the budget (overshooting the real limit is worse
    than leaving room). It is still an estimate; the honest thing is to say so, not fake precision.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, round(max(words * 1.333, chars / 4.0)))


def count_tokens(text: str, model: str | None = None) -> int:
    """Counts the tokens of the text. Exact if ``tiktoken`` is present; heuristic otherwise."""
    enc = _try_tiktoken(model)
    if enc is not None:
        return len(enc.encode(text))
    return _heuristic(text)
