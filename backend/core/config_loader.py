# core/config_loader.py
from __future__ import annotations
import os
import threading
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Path to models.yaml comes from .env
MODELS_YAML = Path(os.getenv("MODELS_CONFIG", "/config/models.yaml"))
print(f"[config_loader] MODELS_YAML path resolved to: {MODELS_YAML}")

_config_lock = threading.Lock()
_config_cache: Optional[Dict[str, Any]] = None

class MissingAPIKey(Exception):
    pass

def _validate_api_keys(cfg: Dict[str, Any]) -> None:
    for name, pdata in (cfg.get("providers") or {}).items():
        env_key = pdata.get("api_key_env")
        if env_key and not os.getenv(env_key):
            print(f"[providers] WARN: Missing API key for '{name}' (env {env_key})")
        else:
            print(f"[providers] OK: Found config for '{name}', env var '{env_key}' set")

def load_config(*, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load and cache the models.yaml defined by MODELS_CONFIG in .env.
    """
    global _config_cache
    with _config_lock:
        if _config_cache is None or force_reload:
            print(f"[load_config] Loading config (force_reload={force_reload})...")
            if not MODELS_YAML.is_file():
                raise FileNotFoundError(
                    f"models.yaml not found at path: {MODELS_YAML}"
                )
            with MODELS_YAML.open("r") as f:
                data = yaml.safe_load(f) or {}
            print(f"[load_config] Raw YAML loaded: {data}")

            if not isinstance(data, dict):
                raise ValueError("models.yaml must parse to a mapping")

            _validate_api_keys(data)
            _config_cache = data
            print(f"[load_config] Cached config: {_config_cache}")
        else:
            print("[load_config] Using cached config")
        return _config_cache

def providers_index(cfg: Dict[str, Any], *, include_secrets: bool = False) -> Dict[str, Any]:
    print(f"[providers_index] Input cfg keys: {list(cfg.keys())}")
    providers = cfg.get("providers") or {}
    print(f"[providers_index] Providers found: {list(providers.keys())}")

    out = {"providers": []}
    for key, pdata in providers.items():
        print(f"[providers_index] Processing provider '{key}': {pdata}")
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
        out["providers"].append(item)

    out["providers"].sort(key=lambda p: p["key"])
    print(f"[providers_index] Final providers list: {out}")
    return out
