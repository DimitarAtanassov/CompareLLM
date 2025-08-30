# routers/providers.py
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from core.config_loader import load_config

router = APIRouter(prefix="/providers", tags=["providers"])

class ProviderOut(BaseModel):
    name: str
    type: str
    base_url: str
    models: List[str] = []
    embedding_models: List[str] = []
    auth_required: bool
    wire: Optional[str] = None  # "openai" | "anthropic" | "gemini" | "ollama" | "cohere"

@router.get("", response_model=Dict[str, List[ProviderOut]])
def list_providers(request: Request):
    # wherever you loaded models.yaml; common patterns:
    # - request.app.state.config["providers"]
    # - request.app.state.providers
    cfg = getattr(request.app.state, "config", None)
    providers_cfg = (cfg or {}).get("providers", {})  # mapping: key -> dict

    print("🔍 [list_providers] Loaded providers config:", providers_cfg)  # DEBUG

    out: List[ProviderOut] = []
    for key, p in providers_cfg.items():
        print(f"➡️ [list_providers] Processing provider '{key}':", p)  # DEBUG

        name = p.get("name") or key
        type_ = p.get("type") or "unknown"
        base_url = p.get("base_url") or ""

        # Normalize: prefer "models", fallback to "chat_models" or "llm_models"
        models = (
            p.get("models")
            or p.get("chat_models")
            or p.get("llm_models")
            or []
        )
        print(f"✅ [list_providers] Models for {key}:", models)  # DEBUG

        embed = p.get("embedding_models") or []
        wire = p.get("wire")
        auth_required = bool(
            p.get("auth_required") is True
            or p.get("requires_api_key") is True
            or p.get("api_key_env")
        )

        out.append(ProviderOut(
            name=name,
            type=type_,
            base_url=base_url,
            models=models,                # ✅ always populated now
            embedding_models=embed,
            auth_required=auth_required,
            wire=wire
        ))

    print("📦 [list_providers] Final output providers:", out)  # DEBUG
    return {"providers": out}


@router.post("/reload")
def reload_providers():
    cfg = load_config(force_reload=True)
    print("♻️ [reload_providers] Reloaded config, providers:", cfg.get("providers"))  # DEBUG
    return {"status": "ok", "providers_count": len((cfg.get("providers") or {}))}


@router.get("/{key}")
def get_provider(
    key: str,
    include_secrets: bool = Query(False, description="Include api_key_env in response"),
):
    cfg = load_config()
    pdata = (cfg.get("providers") or {}).get(key)
    print(f"🔎 [get_provider] Lookup provider '{key}' ->", pdata)  # DEBUG

    if not pdata:
        raise HTTPException(status_code=404, detail=f"Provider '{key}' not found")

    item = {
        "key": key,
        "type": pdata.get("type"),
        "wire": pdata.get("wire"),
        "base_url": pdata.get("base_url"),
        "headers": pdata.get("headers") or {},
        "models": pdata.get("models") or [],
        "embedding_models": pdata.get("embedding_models") or [],
        "requires_api_key": bool(pdata.get("api_key_env")),
    }
    if include_secrets:
        item["api_key_env"] = pdata.get("api_key_env")

    print(f"📤 [get_provider] Returning provider '{key}':", item)  # DEBUG
    return item
