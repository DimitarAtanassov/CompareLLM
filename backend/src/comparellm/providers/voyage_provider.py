"""Adapter for Voyage AI embeddings using the native ``voyageai`` SDK."""

from __future__ import annotations

import voyageai

from comparellm.errors import ProviderError
from comparellm.providers.base import ProviderSpec


class VoyageEmbeddings:
    """Computes embeddings over the Voyage AI embed API."""

    def __init__(self, spec: ProviderSpec, model: str, timeout: float) -> None:
        self.model = model
        self._spec = spec
        self._client = voyageai.AsyncClient(api_key=spec.api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            result = await self._client.embed(texts, model=self.model)
            return [list(vector) for vector in result.embeddings]
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self._spec.key}:{self.model}: {exc}") from exc
