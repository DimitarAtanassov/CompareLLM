"""Shared pytest fixtures: a fully wired app backed by fakes, over ASGI."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from comparellm.domain.chat_service import ChatService
from comparellm.domain.embedding_service import EmbeddingService
from comparellm.infra.prompts.disabled import DisabledPromptCatalog
from comparellm.infra.session.memory import MemorySessionStore
from comparellm.infra.vectorstore.memory import MemoryVectorStore
from comparellm.main import create_app
from comparellm.settings import Settings
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
        self.prompt_catalog = DisabledPromptCatalog()

    async def aclose(self) -> None:
        await self.vector_store.close()
        await self.session_store.close()
        await self.prompt_catalog.close()


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
