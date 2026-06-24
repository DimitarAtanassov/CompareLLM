"""Shared pytest fixtures: a fully wired app backed by fakes, over ASGI."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from app.domain.chat_service import ChatService
from app.domain.embedding_service import EmbeddingService
from app.infra.session.memory import MemorySessionStore
from app.infra.vectorstore.memory import MemoryVectorStore
from app.main import create_app
from app.settings import Settings
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from tests.fakes import FakeRegistry


class FakeContainer:
    """Container double exposing the attributes the API layer reads."""

    def __init__(self) -> None:
        self.settings = Settings(
            sse_heartbeat_seconds=30.0,
            vector_backend="memory",
            session_backend="memory",
            log_json=False,
        )
        self.registry = FakeRegistry()
        self.vector_store = MemoryVectorStore()
        self.session_store = MemorySessionStore()
        self.chat_service = ChatService(self.registry, self.session_store)  # type: ignore[arg-type]
        self.embedding_service = EmbeddingService(self.registry, self.vector_store)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        await self.vector_store.close()
        await self.session_store.close()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    container = FakeContainer()
    app = create_app(container.settings)
    # Pre-seed the container so lifespan uses the fake instead of building a real one.
    app.state.container = container

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client
