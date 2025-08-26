# backend/main.py
import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Master router (chat/embed/health/utils + legacy shims)
from api.router import router as api_router

# Registry + adapters
from providers.registry import ModelRegistry
from providers.adapters.enhanced_chat_adapter import EnhancedChatAdapter
from providers.adapters.embedding_adapter import EmbeddingAdapter

# Services
from services.enhanced_chat_service import EnhancedChatService
from services.embedding_service import EmbeddingService
from services.dataset_service import DatasetService
from services.search_services import SearchService  # or services.search_service if that's your filename

# Storage backend (memory store)
MemoryStorageBackend = None
try:
    from storage.memory_store import MemoryStorageBackend as _MSB
    MemoryStorageBackend = _MSB
except Exception:
    try:
        # Fallback path if your file lives elsewhere but same import path
        from storage.memory_store import MemoryStorageBackend as _MSB
        MemoryStorageBackend = _MSB
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ---------- Startup ----------
    # CORS is middleware so it stays in create_app(); all other state goes here.
    # 1) Registry
    models_yaml_path = os.getenv("MODELS_YAML", "/config/models.yaml")
    app.state.registry = ModelRegistry.from_path(models_yaml_path)

    # 2) Adapters
    app.state.chat_adapter = EnhancedChatAdapter()
    app.state.embedding_adapter = EmbeddingAdapter()

    # 3) Storage
    if MemoryStorageBackend is None:
        raise RuntimeError(
            "No MemoryStorageBackend found. Ensure storage/memory_store.py defines MemoryStorageBackend."
        )
    storage = MemoryStorageBackend()

    # 4) Services (singletons)
    registry = app.state.registry
    chat_adapter = app.state.chat_adapter
    embedding_adapter = app.state.embedding_adapter

    chat = EnhancedChatService(registry=registry, chat_adapter=chat_adapter)
    embedding = EmbeddingService(registry=registry, embedding_adapter=embedding_adapter)
    dataset = DatasetService(registry=registry, embedding_service=embedding, storage=storage)
    search = SearchService(registry=registry, embedding_service=embedding, storage=storage)

    # DI map used by routes
    app.state.services = {
        "chat": chat,
        "embedding": embedding,
        "dataset": dataset,
        "search": search,
    }
    # Back-compat attributes some routers expect
    app.state.embedding_service = embedding
    app.state.search_service = search
    app.state.memory_store = storage

    # Hand control to the app (serving)
    try:
        yield
    finally:
        # ---------- Shutdown ----------
        try:
            closer = getattr(app.state.memory_store, "close", None)
            if callable(closer):
                res = closer()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Playground (Enhanced)",
        version="2.0",
        lifespan=lifespan,   # <- modern FastAPI lifecycle
    )

    # CORS (open for dev; tighten in prod)
    allow = os.getenv("CORS_ALLOW_ORIGINS")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow.split(",") if allow else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount all API routes
    app.include_router(api_router)
    return app


app = create_app()
# Run: uvicorn backend.main:app --reload
