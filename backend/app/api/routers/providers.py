"""Provider inventory endpoints (shape preserved for the existing UI)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import ContainerDep
from app.config.schema import ProviderConfig
from app.errors import NotFoundError

router = APIRouter(prefix="/providers", tags=["providers"])


class ProviderOut(BaseModel):
    name: str
    type: str
    base_url: str
    models: list[str] = []
    embedding_models: list[str] = []
    auth_required: bool
    wire: str | None = None


def _to_out(provider: ProviderConfig) -> ProviderOut:
    return ProviderOut(
        name=provider.display_name,
        type=provider.type,
        base_url=provider.base_url or "",
        models=provider.models,
        embedding_models=provider.embedding_models,
        auth_required=provider.requires_api_key,
        wire=provider.wire,
    )


@router.get("")
def list_providers(container: ContainerDep) -> dict[str, list[ProviderOut]]:
    providers = [_to_out(p) for p in container.config.providers.values()]
    providers.sort(key=lambda p: p.name)
    return {"providers": providers}


@router.post("/reload")
def reload_providers(container: ContainerDep) -> dict[str, object]:
    container.reload_models()
    return {"status": "ok", "providers_count": len(container.config.providers)}


@router.get("/{key}")
def get_provider(key: str, container: ContainerDep) -> ProviderOut:
    provider = container.config.providers.get(key)
    if provider is None:
        raise NotFoundError(f"Provider '{key}' not found")
    return _to_out(provider)
