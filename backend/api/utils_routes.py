# backend/api/utils_routes.py
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, Body
import uuid

from models.enhanced_requests import DatasetUploadRequest
from services.dataset_service import DatasetService

router = APIRouter()

# ---------- Registry + Services ----------
def _get_registry(request: Request):
    reg = getattr(request.app.state, "registry", None)
    if not reg:
        raise HTTPException(status_code=500, detail="Model registry not initialized")
    return reg

def get_dataset_service(request: Request) -> DatasetService:
    services = getattr(request.app.state, "services", {})
    svc = services.get("dataset")
    if not svc:
        raise HTTPException(status_code=500, detail="Dataset service not initialized")
    return svc

# ---------- Providers (for UI) ----------
@router.get("/providers", summary="List providers (array shape for UI)")
async def list_providers(request: Request):
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

# ---------- Helpers ----------
def _ensure_docs(payload: Any) -> List[Dict[str, Any]]:
    """Accepts: array[dict] | array[str] | {documents: array}"""
    if isinstance(payload, list):
        if all(isinstance(x, dict) for x in payload):
            return payload
        if all(isinstance(x, str) for x in payload):
            return [{"text": x} for x in payload]
        raise HTTPException(status_code=400, detail="Array must contain objects or strings")
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        return _ensure_docs(payload["documents"])
    raise HTTPException(status_code=400, detail='Body must be a JSON array or {"documents":[...] }')

def _pick_text_field(docs: List[Dict[str, Any]], explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    for c in ("text", "content", "body", "summary", "title"):
        if any(isinstance(d, dict) and c in d for d in docs):
            return c
    for d in docs:
        if isinstance(d, dict) and d:
            return next(iter(d.keys()))
    raise HTTPException(status_code=400, detail="Could not infer a text field from documents")

def _default_model(request: Request) -> str:
    reg = _get_registry(request)
    emap = getattr(reg, "embedding_map", None)
    if not emap:
        raise HTTPException(status_code=500, detail="Embedding registry not initialized")
    preferred = [m for m in emap if "text-embedding" in m or "embed" in m]
    return preferred[0] if preferred else next(iter(emap))

# ---------- Compat endpoint (JSON only) ----------
@router.post("/upload-dataset", summary="Compat: upload + embed dataset (JSON body)")
async def upload_dataset_compat(
    request: Request,
    body: Any = Body(..., description="Either an array of docs/strings, or {dataset_id, documents, embedding_model, text_field}"),
):
    """
    Accepts:
      1) JSON array (objects OR strings)
      2) Object: { dataset_id, documents, embedding_model, text_field }
    Returns: DatasetUploadResponse
    """
    dsvc = get_dataset_service(request)

    # Case 2: full object with keys
    if isinstance(body, dict) and ("documents" in body or "dataset_id" in body):
        if "documents" not in body:
            raise HTTPException(status_code=400, detail='Missing "documents" in body object')
        docs = _ensure_docs(body["documents"])
        dataset_id = str(body.get("dataset_id") or uuid.uuid4())
        embedding_model = str(body.get("embedding_model") or _default_model(request))
        text_field = _pick_text_field(docs, explicit=body.get("text_field"))
        req_obj = DatasetUploadRequest(
            dataset_id=dataset_id,
            embedding_model=embedding_model,
            text_field=text_field,
            documents=docs,
        )
        return await dsvc.upload_dataset(req_obj)

    # Case 1: plain array
    docs = _ensure_docs(body)
    dataset_id = str(uuid.uuid4())
    embedding_model = _default_model(request)
    text_field = _pick_text_field(docs, explicit=None)
    req_obj = DatasetUploadRequest(
        dataset_id=dataset_id,
        embedding_model=embedding_model,
        text_field=text_field,
        documents=docs,
    )
    return await dsvc.upload_dataset(req_obj)
