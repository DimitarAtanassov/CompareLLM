"""Typed FastAPI dependencies.

Routers depend on these annotated types instead of reaching into
``app.state`` with untyped ``getattr`` calls, keeping the wiring explicit and
type-checked.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.api.container import AppContainer
from app.domain.chat_service import ChatService
from app.domain.embedding_service import EmbeddingService
from app.providers.registry import ProviderRegistry


def get_container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container


ContainerDep = Annotated[AppContainer, Depends(get_container)]


def get_registry(container: ContainerDep) -> ProviderRegistry:
    return container.registry


def get_chat_service(container: ContainerDep) -> ChatService:
    return container.chat_service


def get_embedding_service(container: ContainerDep) -> EmbeddingService:
    return container.embedding_service


RegistryDep = Annotated[ProviderRegistry, Depends(get_registry)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
