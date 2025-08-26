# backend/api/utils_routes.py
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

def _get_registry(request: Request):
    reg = getattr(request.app.state, "registry", None)
    if not reg:
        raise HTTPException(status_code=500, detail="Model registry not initialized")
    return reg

@router.get("/providers", summary="List providers (array shape for UI)")
async def list_providers(request: Request):
    """
    Returns:
      {
        "providers": [
          {
            "name": "openai",
            "type": "openai",
            "base_url": "...",
            "models": [...],
            "embedding_models": [...],
            "auth_required": false,
            "wire": "openai"
          },
          ...
        ]
      }
    """
    reg = _get_registry(request)

    out: List[Dict[str, Any]] = []
    for name, p in (reg.providers or {}).items():
        wire = getattr(p, "wire", None) or getattr(p, "type", "unknown")
        out.append({
            "name": name,
            "type": str(getattr(p, "type", "unknown")),
            "base_url": str(getattr(p, "base_url", "") or ""),
            "models": list(getattr(p, "models", []) or []),
            "embedding_models": list(getattr(p, "embedding_models", []) or []),
            "auth_required": bool(getattr(p, "auth_required", False)),
            "wire": str(wire),
        })
    return {"providers": out}
