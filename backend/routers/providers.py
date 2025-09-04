# routers/providers.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import re
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


# ---------- Helpers for model normalization & vision detection ----------

def _normalize_model_entry(entry: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Accepts either:
      - string model name, or
      - dict-like: { name|model|id, modalities|capabilities|vision|supports_vision, ... }
    Returns: (model_name, metadata_dict)
    """
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("model") or entry.get("id")
        if not isinstance(name, str) or not name:
            # Fallback: stringified dict (shouldn't happen in a good config)
            return str(entry), dict(entry)
        meta = dict(entry)
        return name, meta
    # Unknown shape: stringify
    return str(entry), {}


def _env_regex_list() -> List[re.Pattern]:
    """
    Optional global overrides via env:
      VISION_MODEL_REGEX="^llava.*$,^moondream.*$"
    """
    import os
    raw = os.getenv("VISION_MODEL_REGEX", "").strip()
    if not raw:
        return []
    patterns = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                patterns.append(re.compile(part))
            except re.error:
                pass
    return patterns


# Common heuristics per provider "type"
_OPENAI_VISION = [
    re.compile(r"^gpt-4o($|-)", re.I),
    re.compile(r"^gpt-4\.1($|-)", re.I),
    re.compile(r"^gpt-5($|-)", re.I),          # NEW: gpt-5 family
    re.compile(r"^o[34]($|-)", re.I),          # o3, o4 families
]

_GOOGLE_VISION = [
    re.compile(r"^gemini-1\.5($|-)", re.I),
    re.compile(r"^gemini-pro-vision($|-)", re.I),
    re.compile(r"^gemini-2\.0($|-)", re.I),
    # NEW: 2.5 variants -> flash, flash-lite, pro
    re.compile(r"^gemini-2\.5(-flash(-lite)?|-pro)($|-)", re.I),
]

# Kept for completeness; Anthropic is short-circuited to True below
_ANTHROPIC_VISION = [
    re.compile(r"^claude-3($|-)", re.I),
    re.compile(r"^claude-3\.5($|-)", re.I),
    re.compile(r"^claude-4($|-)", re.I),
    re.compile(r"^claude-4\.1($|-)", re.I),
]

# Ollama & community models are varied; rely on explicit flags + optional env patterns.


def _matches_any(name: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(name) for p in pats)


def _provider_patterns_from_cfg(provider_cfg: Dict[str, Any]) -> List[re.Pattern]:
    """
    models.yaml may include:
      vision_model_patterns:
        - "^llava.*$"
        - "^bakllava.*$"
    """
    pats: List[re.Pattern] = []
    for raw in provider_cfg.get("vision_model_patterns") or []:
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        try:
            pats.append(re.compile(s))
        except re.error:
            pass
    return pats


def _has_explicit_vision_flag(meta: Dict[str, Any]) -> bool:
    # Recognize common capability keys
    if isinstance(meta.get("vision"), bool) and meta["vision"]:
        return True
    if isinstance(meta.get("supports_vision"), bool) and meta["supports_vision"]:
        return True

    # modalities / capabilities arrays
    for key in ("modalities", "capabilities", "modes"):
        val = meta.get(key)
        if isinstance(val, (list, tuple)):
            lowered = [str(x).lower() for x in val]
            if any(x in ("image", "vision", "multimodal") for x in lowered):
                return True
        elif isinstance(val, str):
            s = val.lower()
            if any(tok in s for tok in ("image", "vision", "multimodal")):
                return True
    return False


def _provider_declares_all_vision(provider_cfg: Dict[str, Any]) -> bool:
    """
    Optional provider-level escape hatch in models.yaml:
      vision_all_models: true
    """
    val = provider_cfg.get("vision_all_models")
    return bool(val is True)


def _is_vision_model(provider_type: str, model_name: str, meta: Dict[str, Any], provider_cfg: Dict[str, Any]) -> bool:
    """
    Decide if a model supports image processing.
    Priority:
      1) Explicit flags in model metadata
      2) Provider-level 'vision_all_models: true'
      3) Per-provider regex patterns from config
      4) Built-in heuristics per provider type
      5) Global env regex overrides
    """
    # 1) explicit flags on the model entry
    if _has_explicit_vision_flag(meta):
        return True

    # 2) provider-level "all models support vision"
    if _provider_declares_all_vision(provider_cfg):
        return True

    pt = (provider_type or "").lower()

    # Special rule per requirement: ALL Anthropic models are vision-capable
    if pt == "anthropic":
        return True

    # 3) provider-configured patterns
    if _matches_any(model_name, _provider_patterns_from_cfg(provider_cfg)):
        return True

    # 4) built-in heuristics
    if pt == "openai":
        if _matches_any(model_name, _OPENAI_VISION):
            return True
    elif pt == "google":
        if _matches_any(model_name, _GOOGLE_VISION):
            return True
    elif pt == "anthropic":
        # already returned True above; keep for completeness
        if _matches_any(model_name, _ANTHROPIC_VISION):
            return True
    # cohere / deepseek / cerebras / ollama: be conservative unless explicitly flagged

    # 5) env overrides
    if _matches_any(model_name, _env_regex_list()):
        return True

    return False


# ---------------------- Existing endpoints ----------------------

@router.get("", response_model=Dict[str, List[ProviderOut]])
def list_providers(request: Request):
    cfg = getattr(request.app.state, "config", None)
    providers_cfg = (cfg or {}).get("providers", {})

    out: List[ProviderOut] = []
    for key, p in providers_cfg.items():
        name = p.get("name") or key
        type_ = p.get("type") or "unknown"
        base_url = p.get("base_url") or ""

        # Normalize: prefer "models", fallback to "chat_models" or "llm_models"
        raw_models = p.get("models") or p.get("chat_models") or p.get("llm_models") or []
        # Keep string names for compatibility
        models: List[str] = []
        for entry in raw_models:
            mname, _meta = _normalize_model_entry(entry)
            models.append(mname)

        embed = p.get("embedding_models") or []
        wire = p.get("wire")
        auth_required = bool(
            p.get("auth_required") is True
            or p.get("requires_api_key") is True
            or p.get("api_key_env")
        )

        out.append(
            ProviderOut(
                name=name,
                type=type_,
                base_url=base_url,
                models=models,
                embedding_models=embed,
                auth_required=auth_required,
                wire=wire,
            )
        )

    return {"providers": out}


@router.post("/reload")
def reload_providers():
    cfg = load_config(force_reload=True)
    return {"status": "ok", "providers_count": len((cfg.get("providers") or {}))}


# ---------------------- NEW: vision-only models ----------------------

class VisionProvidersOut(BaseModel):
    providers: List[ProviderOut]


@router.get("/vision", response_model=VisionProvidersOut)
def list_vision_providers(request: Request) -> VisionProvidersOut:
    """
    Returns the same shape as /providers, but with provider.models filtered
    down to only image-capable models. Detection uses:
      - explicit flags in models.yaml entries (vision/supports_vision/modalities)
      - optional provider-level vision_model_patterns (regex list)
      - built-in heuristics for OpenAI/Gemini (+ Anthropic short-circuit)
      - optional env VISION_MODEL_REGEX (comma-separated regexes)
      - optional provider-level 'vision_all_models: true'
    """
    cfg = getattr(request.app.state, "config", None)
    providers_cfg: Dict[str, Any] = (cfg or {}).get("providers", {})  # key -> dict

    out: List[ProviderOut] = []
    for key, p in providers_cfg.items():
        name = p.get("name") or key
        type_ = p.get("type") or "unknown"
        base_url = p.get("base_url") or ""
        wire = p.get("wire")

        # Accept string/dict model entries
        raw_models = p.get("models") or p.get("chat_models") or p.get("llm_models") or []
        vision_models: List[str] = []
        for entry in raw_models:
            mname, meta = _normalize_model_entry(entry)
            if _is_vision_model(type_, mname, meta, p):
                vision_models.append(mname)

        # Only include providers that have at least one vision-capable model
        if not vision_models:
            continue

        auth_required = bool(
            p.get("auth_required") is True
            or p.get("requires_api_key") is True
            or p.get("api_key_env")
        )

        embed = p.get("embedding_models") or []

        out.append(
            ProviderOut(
                name=name,
                type=type_,
                base_url=base_url,
                models=sorted(set(vision_models)),
                embedding_models=embed,  # unchanged; filter if needed in future
                auth_required=auth_required,
                wire=wire,
            )
        )

    return VisionProvidersOut(providers=out)


@router.get("/{key}")
def get_provider(
    key: str,
    include_secrets: bool = Query(False, description="Include api_key_env in response"),
):
    cfg = load_config()
    pdata = (cfg.get("providers") or {}).get(key)

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

    return item
