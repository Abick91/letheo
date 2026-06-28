"""LLM adapter — Letheo's prose injected into a real LLM.

Supports OpenAI and DeepSeek (which is OpenAI-API-compatible by changing ``base_url``).
Local-first remains the runtime default; the LLM is only invoked when the agent explicitly asks
for it.

Usage:

    from letheo_orchestration import Session
    from letheo_orchestration.llm import LLMClient, Memorist

    with Session() as mem:
        # ...seed events...
        agent = Memorist(mem, LLMClient.openai())          # or LLMClient.deepseek()
        reply = agent.ask("user:Xolotl", "how about trail shoes?")
        print(reply)

Credentials are read from the environment (``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY``) unless passed
explicitly to the constructor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .session import Session

if TYPE_CHECKING:
    from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Provider config: a single class, two endpoints.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMConfig:
    """Minimal provider configuration."""

    name: str
    base_url: str | None
    env_var: str
    default_model: str


OPENAI = LLMConfig(
    name="openai",
    base_url=None,                       # SDK uses its default
    env_var="OPENAI_API_KEY",
    default_model="gpt-4o-mini",
)

DEEPSEEK = LLMConfig(
    name="deepseek",
    base_url="https://api.deepseek.com",
    env_var="DEEPSEEK_API_KEY",
    # `deepseek-chat` (modo no-thinking de v4-flash) se retira el 2026-07-24; usamos el nombre nuevo.
    # Para razonamiento, pasar --model deepseek-v4-pro.
    default_model="deepseek-v4-flash",
)


# ──────────────────────────────────────────────────────────────────────────────
# Thin client over the openai SDK (which serves both).
# ──────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """Unified client for OpenAI and DeepSeek (via the ``openai`` SDK)."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The `openai` SDK is missing. Install with: pip install 'letheo-orchestration[llm]' "
                "or pip install openai"
            ) from e

        key = api_key or os.environ.get(config.env_var)
        if not key:
            raise RuntimeError(
                f"No API key found for {config.name}. "
                f"Export {config.env_var} or pass it as api_key=..."
            )

        # `timeout` keeps a stalled response from hanging the process forever (the SDK default is
        # 10 min ≈ "infinite" in a loop of hundreds of calls). `max_retries` retries transient
        # failures (5xx, dropped connection) with the SDK's own exponential backoff.
        kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout, "max_retries": max_retries}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client: OpenAI = OpenAI(**kwargs)
        self._model = model or config.default_model
        self.config = config
        # REAL token accounting (what the API reports), thread-safe for the suite pool.
        import threading
        self._usage_lock = threading.Lock()
        self.usage = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}

    @classmethod
    def openai(cls, *, api_key: str | None = None, model: str | None = None) -> "LLMClient":
        return cls(OPENAI, api_key=api_key, model=model)

    @classmethod
    def deepseek(cls, *, api_key: str | None = None, model: str | None = None) -> "LLMClient":
        return cls(DEEPSEEK, api_key=api_key, model=model)

    @property
    def model(self) -> str:
        return self._model

    def chat(self, system: str, user: str, *, temperature: float = 0.7) -> str:
        """Una sola ronda. Devuelve el texto de la respuesta."""
        text, _ = self.chat_with_usage(system, user, temperature=temperature)
        return text

    def chat_with_usage(self, system: str, user: str, *, temperature: float = 0.7):
        """Like `chat` but also returns the token usage of THIS call: (text, usage_dict).

        `usage_dict` = {prompt, completion, total}. Allows granular accounting (e.g. per arm).
        It also accumulates into the client's total (`self.usage`).
        """
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = {"prompt": 0, "completion": 0, "total": 0, "cache_hit": 0, "cache_miss": 0}
        # Defensive accounting — must never break a call (`usage` may be missing, and in tests with a
        # mocked client the lock doesn't even exist). A failure here is ignored.
        try:
            u = resp.usage
            # DeepSeek separates cached input (repeated prefix, ~1/50 of the price) from non-cached.
            hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
            miss = int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
            prompt = int(u.prompt_tokens)
            if hit == 0 and miss == 0:  # API without cache fields → all at miss price (conservative)
                miss = prompt
            usage = {"prompt": prompt, "completion": int(u.completion_tokens),
                     "total": int(u.total_tokens), "cache_hit": hit, "cache_miss": miss}
            with self._usage_lock:
                for k in ("prompt", "completion", "total", "cache_hit", "cache_miss"):
                    self.usage[k] += usage[k]
                self.usage["calls"] += 1
        except Exception:  # noqa: BLE001
            pass
        return (resp.choices[0].message.content or "").strip(), usage

    # Official DeepSeek prices (USD per 1M tokens): (input_cache_miss, input_cache_hit, output).
    # Source: api-docs.deepseek.com/quick_start/pricing. The v4-pro 75% off expired 2026-05-31.
    PRICES_USD_PER_M = {
        "deepseek-v4-flash": (0.14, 0.0028, 0.28),
        "deepseek-v4-pro":   (1.74, 0.0145, 3.48),
        "deepseek-chat":     (0.14, 0.0028, 0.28),  # = non-thinking mode of v4-flash
    }

    def cost_of(self, cache_miss: int, cache_hit: int, completion: int) -> float:
        """Exact USD cost separating cached/non-cached input and output, per the model's rate."""
        miss_r, hit_r, out_r = self.PRICES_USD_PER_M.get(self._model, (0.14, 0.0028, 0.28))
        return cache_miss / 1e6 * miss_r + cache_hit / 1e6 * hit_r + completion / 1e6 * out_r

    def cost_estimate_usd(self) -> float:
        """Total cost from accumulated `usage` (input hit/miss + output)."""
        return self.cost_of(self.usage["cache_miss"], self.usage["cache_hit"], self.usage["completion"])


# ──────────────────────────────────────────────────────────────────────────────
# Memorist: the agent. Letheo's memory goes to the system prompt; the message to the user.
# ──────────────────────────────────────────────────────────────────────────────


SYSTEM_TEMPLATE = (
    "You are an assistant who deeply knows the person you are talking to. "
    "Your knowledge does NOT come from an event log: it comes from a distilled memory that "
    "summarizes their behavioural trace in a few dense vectors. Trust it as you would trust "
    "what you yourself remember about someone close; do not ask them for context you already have. "
    "Be concrete, personal and concise.\n\n"
    "{memory_block}"
)


@dataclass
class AskResult:
    """Result of an agent interaction."""

    reply: str
    memory_block: str
    model: str
    provider: str
    represented_events: int
    memory_tokens_estimate: int
    extras: dict = field(default_factory=dict)


class Memorist:
    """An agent that remembers using Letheo and reasons with an external LLM."""

    def __init__(
        self,
        session: Session,
        client: LLMClient,
        *,
        token_budget: int = 800,
    ) -> None:
        self.session = session
        self.client = client
        self.token_budget = token_budget

    def ask(
        self,
        subject: str,
        message: str,
        *,
        token_budget: int | None = None,
        temperature: float = 0.7,
    ) -> AskResult:
        """Recalls the subject's essence and answers the message from that knowledge."""
        budget = token_budget or self.token_budget
        evoked = self.session.evoke(subject, token_budget=budget)
        system = SYSTEM_TEMPLATE.format(memory_block=evoked.prose)
        reply = self.client.chat(system, message, temperature=temperature)
        return AskResult(
            reply=reply,
            memory_block=evoked.prose,
            model=self.client.model,
            provider=self.client.config.name,
            represented_events=evoked.context.represented,
            memory_tokens_estimate=evoked.context.token_estimate,
        )
