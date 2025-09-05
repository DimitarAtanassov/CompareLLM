# routers/providers.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import re
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from pydantic import BaseModel

from core.config_loader import load_config
from services.provider_service import ProviderService

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

def get_provider_service(request: Request) -> ProviderService:
    # Use app state config if available for consistency
    cfg = getattr(request.app.state, "config", None)
    return ProviderService(config=cfg)


@router.get("", response_model=Dict[str, List[ProviderOut]])
def list_providers(request: Request, provider_service: ProviderService = Depends(get_provider_service)):
    out = [ProviderOut(**p) for p in provider_service.list_providers()]
    return {"providers": out}


@router.post("/reload")
def reload_providers(provider_service: ProviderService = Depends(get_provider_service)):
    return provider_service.reload_providers()


class VisionProvidersOut(BaseModel):
    providers: List[ProviderOut]


@router.get("/vision", response_model=VisionProvidersOut)
def list_vision_providers(request: Request, provider_service: ProviderService = Depends(get_provider_service)) -> VisionProvidersOut:
    out = [ProviderOut(**p) for p in provider_service.list_vision_providers()]
    return VisionProvidersOut(providers=out)


@router.get("/{key}")
def get_provider(
    key: str,
    include_secrets: bool = Query(False, description="Include api_key_env in response"),
    provider_service: ProviderService = Depends(get_provider_service),
):
    return provider_service.get_provider(key, include_secrets=include_secrets)
