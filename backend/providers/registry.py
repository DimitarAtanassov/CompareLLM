# registry.py (or wherever ModelRegistry lives)

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import yaml
from providers.base import Provider

# ---------------- ModelRegistry ----------------

class ModelRegistry:
    def __init__(self, cfg: Dict[str, Any]):
        self.providers: Dict[str, Provider] = {}
        for pname, p in cfg.get("providers", {}).items():
            prov = Provider(
                name=pname,
                type=p["type"],
                base_url=p["base_url"].rstrip("/"),
                api_key_env=(p.get("api_key_env") or None),
                headers=(p.get("headers") or {}),
                models=(p.get("models") or []),
                embedding_models=(p.get("embedding_models") or []),
            )
            # expose a wire hint; default to provider type
            setattr(prov, "wire", (p.get("wire") or p["type"]))
            self.providers[pname] = prov

        # exact and case-insensitive maps for chat models
        self.model_map: Dict[str, Tuple[Provider, str]] = {}
        self._lc_model_map: Dict[str, Tuple[Provider, str]] = {}
        for prov in self.providers.values():
            for m in prov.models:
                self.model_map[m] = (prov, m)
                self._lc_model_map[m.lower()] = (prov, m)

        # exact and case-insensitive maps for embedding models
        self.embedding_map: Dict[str, Tuple[Provider, str]] = {}
        self._lc_embedding_map: Dict[str, Tuple[Provider, str]] = {}
        for prov in self.providers.values():
            for m in prov.embedding_models:
                self.embedding_map[m] = (prov, m)
                self._lc_embedding_map[m.lower()] = (prov, m)

    @classmethod
    def from_path(cls, path: str) -> "ModelRegistry":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(cfg)

    # --------- helpers the rest of the app can use ---------

    def all_providers(self) -> List[Provider]:
        return list(self.providers.values())

    def get_provider_for_model(self, model: str) -> Provider:
        """Return the Provider that serves this chat model."""
        if model in self.model_map:
            return self.model_map[model][0]
        try:
            return self._lc_model_map[model.lower()][0]
        except KeyError:
            raise ValueError(f"No provider registered for chat model '{model}'")

    def get_provider_for_embedding(self, model: str) -> Provider:
        """Return the Provider that serves this embedding model."""
        if model in self.embedding_map:
            return self.embedding_map[model][0]
        try:
            return self._lc_embedding_map[model.lower()][0]
        except KeyError:
            raise ValueError(f"No provider registered for embedding model '{model}'")

# ---------------- module-level singleton (optional, convenient) ----------------

_registry: Optional[ModelRegistry] = None

def init_registry_from_path(path: str) -> ModelRegistry:
    """Call once at startup."""
    global _registry
    _registry = ModelRegistry.from_path(path)
    return _registry

def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("ModelRegistry is not initialized. Call init_registry_from_path(...) at startup.")
    return _registry

def get_provider_for_model(model: str) -> Provider:
    return get_registry().get_provider_for_model(model)

def get_provider_for_embedding(model: str) -> Provider:
    return get_registry().get_provider_for_embedding(model)
