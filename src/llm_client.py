"""
llm_client.py
=============

A provider-agnostic LLM client with automatic fallback across providers.

Mirrors the pattern from a real CTI pipeline: try a primary provider, and on
failure (rate limit, outage, timeout) fall back to the next one in the chain —
e.g. Mistral -> Gemini -> Claude. The caller gets a uniform interface and
doesn't care which provider actually answered.

Design goals
------------
- One uniform ``complete()`` method regardless of provider.
- Pluggable providers behind a small ABC — add a new one in ~15 lines.
- Built-in retry with exponential backoff per provider.
- Transparent fallback chain; the response says who answered.
- No hard dependency on any vendor SDK at import time (lazy imports), so the
  module is safe to import even if only one SDK is installed.

Example
-------
    from llm_client import LLMClient, MistralProvider, AnthropicProvider

    client = LLMClient(providers=[
        MistralProvider(model="mistral-large-latest"),
        AnthropicProvider(model="claude-sonnet-4-6"),
    ])

    resp = client.complete("Summarize CVE-2024-1234 in one sentence.")
    print(resp.text)
    print(f"answered by: {resp.provider} ({resp.model})")

API keys are read from environment variables by default
(MISTRAL_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger("llm_client")


@dataclass
class LLMResponse:
    """Uniform response object returned regardless of provider."""
    text: str
    provider: str
    model: str
    raw: object = None  # the underlying SDK response, if you need token usage etc.


class LLMProviderError(Exception):
    """Raised when a single provider fails after its own retries."""


class AllProvidersFailedError(Exception):
    """Raised when every provider in the fallback chain has failed."""


class LLMProvider(ABC):
    """Base class for a single LLM provider."""

    name: str = "base"

    def __init__(self, model: str, api_key: str | None = None,
                 max_retries: int = 2, backoff_base: float = 1.0,
                 timeout: float = 60.0) -> None:
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout

    @abstractmethod
    def _call(self, prompt: str, system: str | None, **kwargs) -> LLMResponse:
        """Provider-specific single API call. Raise on failure."""
        raise NotImplementedError

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        """Call the provider with retry + exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call(prompt, system, **kwargs)
            except Exception as exc:  # noqa: BLE001 - we re-raise as provider error
                last_exc = exc
                wait = self.backoff_base * (2 ** attempt)
                logger.warning("[%s] attempt %d/%d failed: %s",
                               self.name, attempt + 1, self.max_retries + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise LLMProviderError(f"{self.name} failed after "
                               f"{self.max_retries + 1} attempt(s): {last_exc}")


class MistralProvider(LLMProvider):
    name = "mistral"

    def _call(self, prompt: str, system: str | None, **kwargs) -> LLMResponse:
        from mistralai import Mistral  # lazy import
        key = self.api_key or os.environ["MISTRAL_API_KEY"]
        client = Mistral(api_key=key)
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.complete(model=self.model, messages=messages, **kwargs)
        return LLMResponse(
            text=resp.choices[0].message.content,
            provider=self.name, model=self.model, raw=resp,
        )


class GeminiProvider(LLMProvider):
    name = "gemini"

    def _call(self, prompt: str, system: str | None, **kwargs) -> LLMResponse:
        import google.generativeai as genai  # lazy import
        key = self.api_key or os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=key)
        model = genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(prompt, **kwargs)
        return LLMResponse(
            text=resp.text, provider=self.name, model=self.model, raw=resp,
        )


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def _call(self, prompt: str, system: str | None, **kwargs) -> LLMResponse:
        import anthropic  # lazy import
        key = self.api_key or os.environ["ANTHROPIC_API_KEY"]
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(text=text, provider=self.name, model=self.model, raw=resp)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def _call(self, prompt: str, system: str | None, **kwargs) -> LLMResponse:
        from openai import OpenAI  # lazy import
        key = self.api_key or os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=key)
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model, messages=messages, **kwargs)
        return LLMResponse(
            text=resp.choices[0].message.content,
            provider=self.name, model=self.model, raw=resp,
        )


class LLMClient:
    """
    Orchestrates a fallback chain of providers.

    On ``complete()``, tries each provider in order. The first to succeed wins.
    If all fail, raises AllProvidersFailedError with the collected errors.
    """

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("LLMClient needs at least one provider.")
        self.providers = providers

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        errors: list[str] = []
        for provider in self.providers:
            try:
                logger.info("Trying provider: %s", provider.name)
                return provider.complete(prompt, system, **kwargs)
            except LLMProviderError as exc:
                errors.append(str(exc))
                logger.warning("Falling back from %s", provider.name)
        raise AllProvidersFailedError(
            "All providers failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Demonstration with a fake provider so it runs with no API keys/SDKs.
    class _Flaky(LLMProvider):
        name = "flaky"

        def _call(self, prompt, system, **kwargs):
            raise RuntimeError("simulated outage")

    class _Stub(LLMProvider):
        name = "stub"

        def _call(self, prompt, system, **kwargs):
            return LLMResponse(text=f"[stub answer to: {prompt[:40]}...]",
                               provider=self.name, model=self.model)

    client = LLMClient(providers=[
        _Flaky(model="x", max_retries=1, backoff_base=0.01),
        _Stub(model="stub-1"),
    ])
    out = client.complete("Summarize CVE-2024-1234 in one sentence.")
    print(f"\nanswered by {out.provider}: {out.text}")
