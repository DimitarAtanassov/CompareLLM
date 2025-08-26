# app/backend/core/dependencies.py
from functools import lru_cache

from fastapi import Depends

from config.settings import Settings, get_settings
from providers.registry import ModelRegistry
from providers.adapters.enhanced_chat_adapter import ChatAdapter, EnhancedChatAdapter
from providers.adapters.embedding_adapter import EmbeddingAdapter
from services.enhanced_chat_service import ChatService, EnhancedChatService
from services.embedding_service import EmbeddingService
from services.dataset_service import DatasetService
from services.search_services import SearchService
from storage.memory_store import MemoryStorageBackend


# Remove @lru_cache() from functions that take Settings objects
def get_model_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    """Get the model registry singleton."""
    return ModelRegistry.from_path(settings.models_config_path)


@lru_cache()
def get_chat_adapter(
    settings: Settings = Depends(get_settings)
) -> EnhancedChatAdapter:
    """Get the chat adapter singleton."""
    return EnhancedChatAdapter()


@lru_cache()
def get_embedding_adapter() -> EmbeddingAdapter:
    """Get the embedding adapter singleton."""
    return EmbeddingAdapter()


@lru_cache()
def get_storage_backend() -> MemoryStorageBackend:
    """Get the storage backend singleton."""
    return MemoryStorageBackend()


def get_chat_service(
    registry: ModelRegistry = Depends(get_model_registry),
    chat_adapter: EnhancedChatAdapter = Depends(get_chat_adapter)
) -> EnhancedChatService:
    """Get chat service with dependencies."""
    return EnhancedChatService(registry, chat_adapter)


def get_embedding_service(
    registry: ModelRegistry = Depends(get_model_registry),
    embedding_adapter: EmbeddingAdapter = Depends(get_embedding_adapter)
) -> EmbeddingService:
    """Get embedding service with dependencies."""
    return EmbeddingService(registry, embedding_adapter)


def get_dataset_service(
    registry: ModelRegistry = Depends(get_model_registry),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    storage: MemoryStorageBackend = Depends(get_storage_backend)
) -> DatasetService:
    """Get dataset service with dependencies."""
    return DatasetService(registry, embedding_service, storage)


def get_search_service(
    registry: ModelRegistry = Depends(get_model_registry),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    storage: MemoryStorageBackend = Depends(get_storage_backend)
) -> SearchService:
    """Get search service with dependencies."""
    return SearchService(registry, embedding_service, storage)