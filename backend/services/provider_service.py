from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import re
from fastapi import HTTPException
from core.config_loader import load_config

class ProviderService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.providers_cfg = self.config.get("providers", {})

    def list_providers(self) -> List[Dict[str, Any]]:
        out = []
        for key, p in self.providers_cfg.items():
            name = p.get("name") or key
            type_ = p.get("type") or "unknown"
            base_url = p.get("base_url") or ""
            raw_models = p.get("models") or p.get("chat_models") or p.get("llm_models") or []
            models = [self._normalize_model_entry(entry)[0] for entry in raw_models]
            embed = p.get("embedding_models") or []
            wire = p.get("wire")
            auth_required = bool(
                p.get("auth_required") is True
                or p.get("requires_api_key") is True
                or p.get("api_key_env")
            )
            out.append({
                "name": name,
                "type": type_,
                "base_url": base_url,
                "models": models,
                "embedding_models": embed,
                "auth_required": auth_required,
                "wire": wire,
            })
        return out

    def reload_providers(self) -> Dict[str, Any]:
        self.config = load_config(force_reload=True)
        self.providers_cfg = self.config.get("providers", {})
        return {"status": "ok", "providers_count": len(self.providers_cfg)}

    def get_provider(self, key: str, include_secrets: bool = False) -> Dict[str, Any]:
        pdata = self.providers_cfg.get(key)
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

    def list_vision_providers(self) -> List[Dict[str, Any]]:
        out = []
        for key, p in self.providers_cfg.items():
            name = p.get("name") or key
            type_ = p.get("type") or "unknown"
            base_url = p.get("base_url") or ""
            wire = p.get("wire")
            raw_models = p.get("models") or p.get("chat_models") or p.get("llm_models") or []
            vision_models = []
            for entry in raw_models:
                mname, meta = self._normalize_model_entry(entry)
                if self._is_vision_model(type_, mname, meta, p):
                    vision_models.append(mname)
            if not vision_models:
                continue
            auth_required = bool(
                p.get("auth_required") is True
                or p.get("requires_api_key") is True
                or p.get("api_key_env")
            )
            embed = p.get("embedding_models") or []
            out.append({
                "name": name,
                "type": type_,
                "base_url": base_url,
                "models": sorted(set(vision_models)),
                "embedding_models": embed,
                "auth_required": auth_required,
                "wire": wire,
            })
        return out

    # --- Helper methods (copied and adapted from router) ---
    def _normalize_model_entry(self, entry: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(entry, str):
            return entry, {}
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model") or entry.get("id")
            if not isinstance(name, str) or not name:
                return str(entry), dict(entry)
            meta = dict(entry)
            return name, meta
        return str(entry), {}

    def _has_explicit_vision_flag(self, meta: Dict[str, Any]) -> bool:
        if isinstance(meta.get("vision"), bool) and meta["vision"]:
            return True
        if isinstance(meta.get("supports_vision"), bool) and meta["supports_vision"]:
            return True
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

    def _provider_declares_all_vision(self, provider_cfg: Dict[str, Any]) -> bool:
        val = provider_cfg.get("vision_all_models")
        return bool(val is True)

    def _matches_any(self, name: str, pats: List[re.Pattern]) -> bool:
        return any(p.search(name) for p in pats)

    def _provider_patterns_from_cfg(self, provider_cfg: Dict[str, Any]) -> List[re.Pattern]:
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

    def _env_regex_list(self) -> List[re.Pattern]:
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

    def _is_vision_model(self, provider_type: str, model_name: str, meta: Dict[str, Any], provider_cfg: Dict[str, Any]) -> bool:
        if self._has_explicit_vision_flag(meta):
            return True
        if self._provider_declares_all_vision(provider_cfg):
            return True
        pt = (provider_type or "").lower()
        if pt == "anthropic":
            return True
        # Built-in heuristics
        _OPENAI_VISION = [
            re.compile(r"^gpt-4o($|-)", re.I),
            re.compile(r"^gpt-4\\.1($|-)", re.I),
            re.compile(r"^gpt-5($|-)", re.I),
            re.compile(r"^o[34]($|-)", re.I),
        ]
        _GOOGLE_VISION = [
            re.compile(r"^gemini-1\\.5($|-)", re.I),
            re.compile(r"^gemini-pro-vision($|-)", re.I),
            re.compile(r"^gemini-2\\.0($|-)", re.I),
            re.compile(r"^gemini-2\\.5(-flash(-lite)?|-pro)($|-)", re.I),
        ]
        _ANTHROPIC_VISION = [
            re.compile(r"^claude-3($|-)", re.I),
            re.compile(r"^claude-3\\.5($|-)", re.I),
            re.compile(r"^claude-4($|-)", re.I),
            re.compile(r"^claude-4\\.1($|-)", re.I),
        ]
        if self._matches_any(model_name, self._provider_patterns_from_cfg(provider_cfg)):
            return True
        if pt == "openai":
            if self._matches_any(model_name, _OPENAI_VISION):
                return True
        elif pt == "google":
            if self._matches_any(model_name, _GOOGLE_VISION):
                return True
        elif pt == "anthropic":
            if self._matches_any(model_name, _ANTHROPIC_VISION):
                return True
        if self._matches_any(model_name, self._env_regex_list()):
            return True
        return False
