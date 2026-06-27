"""Adapter for Anthropic Claude models using the native ``anthropic`` SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from comparellm.domain.models import ChatMessage, GenerationParams
from comparellm.errors import ProviderError
from comparellm.providers.base import ProviderSpec, split_system
from comparellm.providers.params import anthropic_kwargs


class AnthropicChat:
    """Streams chat completions over the Anthropic Messages API."""

    def __init__(
        self, spec: ProviderSpec, model: str, timeout: float, default_max_tokens: int
    ) -> None:
        self.model = model
        self._spec = spec
        self._default_max_tokens = default_max_tokens
        self._client = AsyncAnthropic(
            api_key=spec.api_key,
            default_headers=spec.headers or None,
            timeout=timeout,
        )

    async def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        system, conversation = split_system(messages)
        payload = [{"role": m.role, "content": m.content} for m in conversation]
        kwargs: dict[str, Any] = anthropic_kwargs(params, self._default_max_tokens)
        if system:
            kwargs["system"] = system
        try:
            async with self._client.messages.stream(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc
