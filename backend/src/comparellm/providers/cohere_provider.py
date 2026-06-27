"""Adapter for Cohere Command (chat) and Embed models using the ``cohere`` v2 SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator

import cohere

from comparellm.domain.models import ChatMessage, GenerationParams
from comparellm.errors import ProviderError
from comparellm.providers.base import ProviderSpec
from comparellm.providers.params import cohere_kwargs


class CohereChat:
    """Streams chat completions over the Cohere v2 chat API."""

    def __init__(self, spec: ProviderSpec, model: str, timeout: float) -> None:
        self.model = model
        self._spec = spec
        self._client = cohere.AsyncClientV2(api_key=spec.api_key, timeout=timeout)

    async def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        # Cohere v2 supports the system role natively in the messages array.
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            stream = self._client.chat_stream(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
                **cohere_kwargs(params),
            )
            async for event in stream:
                if event.type == "content-delta":
                    text = event.delta.message.content.text  # type: ignore[union-attr]
                    if text:
                        yield text
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc


class CohereEmbeddings:
    """Computes embeddings over the Cohere v2 embed API."""

    def __init__(self, spec: ProviderSpec, model: str, timeout: float) -> None:
        self.model = model
        self._spec = spec
        self._client = cohere.AsyncClientV2(api_key=spec.api_key, timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embed(
                model=self.model,
                texts=texts,
                input_type="search_document",
                embedding_types=["float"],
            )
            return list(response.embeddings.float_ or [])
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc
