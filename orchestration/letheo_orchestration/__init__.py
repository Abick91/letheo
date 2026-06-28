"""letheo_orchestration · High-level Python layer over the Cognitive Runtime.

Provides:
- ``Session``: ergonomic API with a context manager, executable MQL and prose for the LLM.
- ``to_prose``: turns a ``CompressedContext`` into a narrative block ready to inject
  into an LLM prompt (Claude, GPT, llama, etc.).

Local-first: no network or external SDKs required. Integration with a concrete LLM happens outside.
"""
from .session import Session, EvokeResult
from .prose import to_prose, ArcReading, read_arc
from .tokens import count_tokens, count_tokens_method

__all__ = [
    "Session", "EvokeResult", "to_prose", "ArcReading", "read_arc",
    "count_tokens", "count_tokens_method",
    # ``llm`` is imported lazily: requires the `[llm]` extra.
]


def __getattr__(name: str):
    if name in {"LLMClient", "Memorist", "AskResult", "OPENAI", "DEEPSEEK"}:
        from . import llm
        return getattr(llm, name)
    raise AttributeError(name)
