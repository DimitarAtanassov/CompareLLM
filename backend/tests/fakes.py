"""Deterministic, network-free provider fakes for tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.domain.models import ChatMessage, GenerationParams

_DIM = 16


class FakeChatProvider:
    """Streams a fixed script of tokens derived from the model name."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        last = messages[-1].content if messages else ""
        for token in ["echo:", f" {self.model}", f" <- {last}"]:
            yield token


class FakeEmbeddingProvider:
    """Deterministic character-histogram embedding (identical text => identical vector)."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * _DIM
            for char in text:
                vector[ord(char) % _DIM] += 1.0
            vectors.append(vector)
        return vectors


class FakeRegistry:
    """Stand-in for ProviderRegistry exposing only what the services use."""

    def __init__(self) -> None:
        self._chat = {"fake:alpha", "fake:beta"}
        self._embed = {"fake:embed", "fake:embed2"}

    def chat_models(self) -> list[str]:
        return sorted(self._chat)

    def embedding_models(self) -> list[str]:
        return sorted(self._embed)

    def has_chat(self, target: str) -> bool:
        return target in self._chat

    def has_embedding(self, target: str) -> bool:
        return target in self._embed

    def get_chat(self, target: str) -> FakeChatProvider:
        _, model = target.split(":", 1)
        return FakeChatProvider(model)

    def get_embedder(self, target: str) -> FakeEmbeddingProvider:
        return FakeEmbeddingProvider(target)
