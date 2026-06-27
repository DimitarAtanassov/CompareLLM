"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

from comparellm.api.deps import RegistryDep

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up and serving."""
    return {"status": "ok"}


@router.get("/readyz")
def readiness(registry: RegistryDep) -> dict[str, object]:
    """Readiness: configuration loaded and at least one model is available."""
    chat_models = registry.chat_models()
    embedding_models = registry.embedding_models()
    ready = bool(chat_models or embedding_models)
    return {
        "ready": ready,
        "chat_models": len(chat_models),
        "embedding_models": len(embedding_models),
    }
