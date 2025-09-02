from __future__ import annotations
from typing import Dict, Any, Tuple
from fastapi import HTTPException

def _log(msg: str) -> None:
    print(f"[EmbeddingUtil] {msg}")

def resolve_embedding_key_from_model(cfg: Dict[str, Any], model_name: str) -> str:
    providers = cfg.get("providers") or {}
    for pkey, pcfg in providers.items():
        if model_name in (pcfg.get("embedding_models") or []):
            ek = f"{pkey}:{model_name}"
            _log(f"Resolved embedding key for model '{model_name}' -> {ek}")
            return ek
    _log(f"ERROR: Embedding model '{model_name}' not registered in models.yaml")
    raise HTTPException(status_code=404, detail=f"Embedding model '{model_name}' not configured in models.yaml")

def store_id(dataset_id: str, embedding_key: str) -> str:
    sid = f"{dataset_id}::{embedding_key}"
    _log(f"Composed store_id -> {sid}")
    return sid

def parse_store_id(s: str) -> Tuple[str, str]:
    try:
        ds, ek = s.split("::", 1)
        _log(f"Parsed store_id='{s}' -> dataset='{ds}', emb='{ek}'")
        return ds, ek
    except Exception:
        _log(f"WARN: Failed to parse store_id='{s}'")
        return s, ""
