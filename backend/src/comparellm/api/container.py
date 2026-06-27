"""Composition root.

Wires settings, config, the provider registry, persistence backends, and the
domain services into a single container that lives on ``app.state``. Centralizing
construction here keeps wiring out of routers and makes the dependency graph
explicit and testable.
"""

from __future__ import annotations

from comparellm.config import ModelsConfig, load_models_config
from comparellm.domain.chat_service import ChatService
from comparellm.domain.embedding_service import EmbeddingService
from comparellm.infra.prompts import PromptCatalog, build_prompt_catalog
from comparellm.infra.session import SessionStore, build_session_store
from comparellm.infra.vectorstore import VectorStore, build_vector_store
from comparellm.log import get_logger
from comparellm.providers.registry import ProviderRegistry
from comparellm.settings import Settings

log = get_logger(__name__)


class AppContainer:
    """Holds the long-lived application services and their dependencies."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config: ModelsConfig = load_models_config(settings.models_config)
        self.registry = ProviderRegistry(self.config, settings)
        self.vector_store: VectorStore = build_vector_store(settings)
        self.session_store: SessionStore = build_session_store(settings)
        self.prompt_catalog: PromptCatalog = build_prompt_catalog(settings)
        self.chat_service = ChatService(self.registry, self.session_store)
        self.embedding_service = EmbeddingService(self.registry, self.vector_store)
        log.info(
            "container_initialized",
            vector_backend=settings.vector_backend,
            session_backend=settings.session_backend,
            prompts_enabled=settings.floating_prompts_url is not None,
        )

    def reload_models(self) -> None:
        """Reload ``models.yaml`` and rebuild the registry + services.

        Persistence backends (vector/session stores) are preserved so indexed
        data and live conversations survive a config reload.
        """
        self.config = load_models_config(self.settings.models_config)
        self.registry = ProviderRegistry(self.config, self.settings)
        self.chat_service = ChatService(self.registry, self.session_store)
        self.embedding_service = EmbeddingService(self.registry, self.vector_store)
        log.info("container_reloaded", providers=sorted(self.config.providers))

    async def aclose(self) -> None:
        await self.vector_store.close()
        await self.session_store.close()
        await self.prompt_catalog.close()
        log.info("container_closed")
