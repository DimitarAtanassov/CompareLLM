"""Adapter for Google Gemini models using the native ``google-genai`` SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from app.domain.models import ChatMessage, GenerationParams
from app.errors import ProviderError
from app.providers.base import ProviderSpec, split_system
from app.providers.params import gemini_config_kwargs


def _to_contents(messages: list[ChatMessage]) -> list[types.Content]:
    """Map domain messages to Gemini ``Content`` objects (roles: user/model)."""
    contents: list[types.Content] = []
    for message in messages:
        role = "user" if message.role == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=message.content)])
        )
    return contents


class GeminiChat:
    """Streams chat completions over the Gemini ``generate_content`` API."""

    def __init__(self, spec: ProviderSpec, model: str, timeout: float) -> None:
        self.model = model
        self._spec = spec
        self._client = genai.Client(api_key=spec.api_key)

    def _build_config(
        self, system: str | None, params: GenerationParams
    ) -> types.GenerateContentConfig:
        config_kwargs: dict[str, Any] = gemini_config_kwargs(params)
        if system:
            config_kwargs["system_instruction"] = system
        if params.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=params.thinking_budget
            )
        return types.GenerateContentConfig(**config_kwargs)

    async def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        system, conversation = split_system(messages)
        config = self._build_config(system, params)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=_to_contents(conversation),  # type: ignore[arg-type]
                config=config,
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc


class GeminiEmbeddings:
    """Computes embeddings over the Gemini ``embed_content`` API."""

    def __init__(self, spec: ProviderSpec, model: str, timeout: float) -> None:
        self.model = model
        self._spec = spec
        self._client = genai.Client(api_key=spec.api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            result = await self._client.aio.models.embed_content(
                model=self.model,
                contents=texts,  # type: ignore[arg-type]
            )
            return [list(embedding.values or []) for embedding in result.embeddings or []]
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc
