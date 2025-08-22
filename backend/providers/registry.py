from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import yaml
from providers.base import Provider


class ModelRegistry:
    def __init__(self, cfg: Dict[str, Any]):
        self.providers: Dict[str, Provider] = {}
        for pname, p in cfg.get("providers", {}).items():
            self.providers[pname] = Provider(
                name=pname,
                type=p["type"],
                base_url=p["base_url"].rstrip("/"),
                api_key_env=(p.get("api_key_env") or None),
                headers=(p.get("headers") or {}),
                models=(p.get("models") or []),
                embedding_models=(p.get("embedding_models") or []),
            )
        
        # map model -> (provider, model_name)
        self.model_map: Dict[str, Tuple[Provider, str]] = {}
        for prov in self.providers.values():
            for m in prov.models:
                self.model_map[m] = (prov, m)
        
        # map embedding_model -> (provider, model_name)
        self.embedding_map: Dict[str, Tuple[Provider, str]] = {}
        for prov in self.providers.values():
            for m in prov.embedding_models:
                self.embedding_map[m] = (prov, m)

    @classmethod
    def from_path(cls, path: str) -> "ModelRegistry":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(cfg)