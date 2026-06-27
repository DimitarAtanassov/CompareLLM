"""Adapter for OpenAI and all OpenAI-compatible providers.

Covers ``openai``, ``deepseek``, ``cerebras``, ``ollama`` (via its ``/v1``
OpenAI-compatible endpoint), and any provider declaring ``wire: openai``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from comparellm.domain.models import ChatMessage, GenerationParams
from comparellm.errors import ProviderError
from comparellm.providers.base import ProviderSpec
from comparellm.providers.params import openai_kwargs


def _make_client(spec: ProviderSpec, timeout: float) -> AsyncOpenAI:
    return AsyncOpenAI(
        # Local gateways (Ollama) need no key; the SDK still requires a non-empty string.
        api_key=spec.api_key or "not-required",
        base_url=spec.base_url,
        default_headers=spec.headers or None,
        timeout=timeout,
    )


class OpenAICompatChat:
    """Streams chat completions over the OpenAI chat.completions API."""

    def __init__(self, spec: ProviderSpec, model: str, timeout: float) -> None:
        self.model = model
        self._spec = spec
        self._client = _make_client(spec, timeout)

    async def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
                stream=True,
                **openai_kwargs(params),
            )
            # stream=True makes this an AsyncStream; the SDK's overload union
            # cannot be narrowed statically through **kwargs.
            async for chunk in stream:  # type: ignore[union-attr]
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc


class OpenAICompatEmbeddings:
    """Computes embeddings over the OpenAI embeddings API."""

    def __init__(self, spec: ProviderSpec, model: str, timeout: float) -> None:
        self.model = model
        self._spec = spec
        self._client = _make_client(spec, timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc
