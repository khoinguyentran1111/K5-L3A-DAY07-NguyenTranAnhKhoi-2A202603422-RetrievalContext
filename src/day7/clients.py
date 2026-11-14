"""OpenRouter clients (embeddings + chat) with retries and usage/cost accounting. No mock fallback."""
import time
from typing import Any

import openai
from openai import OpenAI

from . import config

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)


def _client() -> OpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
    return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY, max_retries=0, timeout=90)


def _with_retries(fn, retries: int = 3):
    for attempt in range(retries + 1):
        try:
            return fn()
        except RETRYABLE:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))


class Usage:
    """Accumulates calls, tokens and USD cost reported by OpenRouter (usage.cost)."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def add(self, usage: Any) -> dict[str, Any]:
        self.calls += 1
        rec = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        if usage is not None:
            rec["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
            rec["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
            rec["cost_usd"] = float(getattr(usage, "cost", 0) or 0)
        self.prompt_tokens += rec["prompt_tokens"]
        self.completion_tokens += rec["completion_tokens"]
        self.cost_usd += rec["cost_usd"]
        return rec

    def snapshot(self) -> dict[str, Any]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cost_usd": round(self.cost_usd, 6)}


class Embedder:
    """Same model and call as Day 6 indexing (text-embedding-3-small needs no query prefix)."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.EMBEDDING_MODEL
        if not self.model:
            raise RuntimeError("EMBEDDING_MODEL is missing in .env")
        self.client = _client()
        self.usage = Usage()
        self._cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        resp = _with_retries(lambda: self.client.embeddings.create(model=self.model, input=[text]))
        self.usage.add(resp.usage)
        vec = resp.data[0].embedding
        if len(vec) != config.EMBEDDING_DIMENSIONS:
            raise RuntimeError(f"query embedding has {len(vec)} dims, index expects {config.EMBEDDING_DIMENSIONS}")
        self._cache[text] = vec
        return vec

    def clear_cache(self) -> None:
        self._cache.clear()


class Chat:
    """Thin wrapper over chat.completions; returns the raw message plus a usage record."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.AGENT_MODEL
        self.client = _client()
        self.usage = Usage()

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                 tool_choice: Any = None, temperature: float = 0.0, max_tokens: int = 1200,
                 response_format: dict[str, Any] | None = None,
                 reasoning: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        # The model thinks by default; hidden reasoning tokens can eat max_tokens and leave content empty.
        # Default: reasoning disabled (cheaper, more repeatable); configs may enable it explicitly.
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                  "max_tokens": max_tokens,
                                  "extra_body": {"usage": {"include": True},
                                                 "reasoning": reasoning or {"enabled": False}}}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        started = time.perf_counter()
        resp = _with_retries(lambda: self.client.chat.completions.create(**kwargs))
        rec = self.usage.add(resp.usage)
        rec["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if not resp.choices:
            raise RuntimeError(f"model returned no choices: {resp}")
        choice = resp.choices[0]
        rec["finish_reason"] = choice.finish_reason
        if not choice.message.content and not choice.message.tool_calls:
            raise RuntimeError(f"model returned empty content (finish_reason={choice.finish_reason})")
        return choice.message, rec
