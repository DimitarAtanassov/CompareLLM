# routers/vision.py
from __future__ import annotations
import asyncio
import base64
import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from graphs.factory import build_single_model_graph, build_multi_model_graph
from core.model_factory import resolve_and_init_from_registry

router = APIRouter(prefix="/vision", tags=["vision"])

# ---- SSE helpers (same shape as /langgraph routes) ----
def _sse_event(data: Dict[str, Any], event: Optional[str] = None) -> bytes:
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")

def _sse_comment(comment: str = "") -> bytes:
    return (f":{comment}\n\n").encode("utf-8")

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# ---- Message/content helpers ----
def _is_png_or_jpeg(mime: str) -> bool:
    m = (mime or "").lower()
    return m in ("image/png", "image/jpeg", "image/jpg")

def _b64(s: bytes) -> str:
    return base64.b64encode(s).decode("ascii")

def _b64_data_uri(mime: str, b: bytes) -> str:
    return f"data:{mime};base64,{_b64(b)}"

def _anthropic_part(mime: str, b: bytes) -> dict:
    # anthropic expects {"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": _b64(b)}}

def _image_url_part(data_uri: str) -> dict:
    # openai / google / cohere (LC adapters) accept {"type":"image_url","image_url": {"url": "data:..."}}
    return {"type": "image_url", "image_url": {"url": data_uri}}

def _system_msg_if(system_text: Optional[str]) -> List[BaseMessage]:
    return [SystemMessage(content=system_text)] if system_text else []

def _build_vision_human_message(
    provider_type: str,       # e.g., "openai", "anthropic", "google", "cohere", "cerebras"
    prompt_text: Optional[str],
    mime: str,
    raw: bytes,
) -> HumanMessage:
    """
    Create a provider-aware HumanMessage with multimodal content:
    - anthropic: use base64 image part
    - openai/cerebras/deepseek(gpt-wire)/google/cohere: use image_url data URI
    """
    parts: List[dict] = []
    if prompt_text:
        parts.append({"type": "text", "text": prompt_text})

    ptype = (provider_type or "").lower()
    if ptype == "anthropic":
        parts.append(_anthropic_part(mime, raw))
    else:
        parts.append(_image_url_part(_b64_data_uri(mime, raw)))

    return HumanMessage(content=parts)

# --- registry + provider helpers
def _registry(request: Request):
    reg = getattr(request.app.state, "registry", None)
    if reg is None:
        raise HTTPException(500, "Model registry not initialized")
    return reg

def _memory_backend(request: Request):
    return getattr(request.app.state, "graph_memory", None)

def _provider_type_from_wire(reg, wire: str) -> Tuple[str, str, str]:
    """Return (provider_key, model_name, provider_type) from 'prov:model'."""
    try:
        pkey, model_name = wire.split(":", 1)
    except ValueError:
        raise HTTPException(400, "wire must be 'provider:model'")
    ptype = (reg.provider_type(pkey) or "").lower()
    return pkey, model_name, ptype

def _choose_upload(image: Optional[UploadFile], file_: Optional[UploadFile]) -> UploadFile:
    if file_ is not None:
        return file_
    if image is not None:
        return image
    raise HTTPException(400, "Missing file upload: expected form field 'file'")

# ---- Output extraction helpers ----
def _extract_text_from_output(output: Any) -> str:
    # Handles AIMessage / BaseMessage / simple strings
    if isinstance(output, AIMessage):
        c = output.content
        if isinstance(c, str):
            return c
        # LC can return list parts; extract any text parts if present
        if isinstance(c, list):
            buf: List[str] = []
            for part in c:
                if isinstance(part, dict):
                    if part.get("type") in ("text", "raw"):
                        t = part.get("text") or part.get("value") or ""
                        if isinstance(t, str):
                            buf.append(t)
            if buf:
                return "".join(buf)
        # Fallback to .text attr if adapter exposes it
        t = getattr(output, "text", None)
        if isinstance(t, str):
            return t
        return ""
    # LangChain sometimes returns a dict-like; try common fields
    if isinstance(output, dict):
        for k in ("text", "message", "content"):
            v = output.get(k)
            if isinstance(v, str):
                return v
    if isinstance(output, str):
        return output
    # Generic BaseMessage
    if hasattr(output, "content") and isinstance(getattr(output, "content"), str):
        return getattr(output, "content")
    return ""

def _extract_image_from_output(output: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Tries to find an image payload in common chat-model outputs.
    Returns (image_base64, image_mime, image_url).
    """
    # Look inside AIMessage.content if it's a list of parts
    parts: List[Any] = []
    if isinstance(output, AIMessage):
        if isinstance(output.content, list):
            parts = output.content
    elif isinstance(output, dict) and isinstance(output.get("content"), list):
        parts = output["content"]  # type: ignore

    # Common part shapes
    for part in parts:
        if not isinstance(part, dict):
            continue
        # OpenAI-ish image_url
        if part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if isinstance(url, str) and url:
                return None, None, url
        # Base64 image
        if part.get("type") in ("image", "image_base64"):
            src = part.get("source") or {}
            data = src.get("data") or part.get("data")
            mime = src.get("media_type") or part.get("mime") or "image/png"
            if isinstance(data, str) and data:
                # Could be already base64 or a data URI
                if data.startswith("data:"):
                    # data URI
                    try:
                        header, b64 = data.split(",", 1)
                        mime_guess = header.split(";")[0].split(":", 1)[1] if ":" in header else "image/png"
                        return b64, mime_guess, data
                    except Exception:
                        pass
                return data, mime, None

    # Some providers put base64 string in additional_kwargs
    if isinstance(output, AIMessage):
        ak = getattr(output, "additional_kwargs", {}) or {}
        # Non-standard; check a few common keys
        for key in ("image_base64", "image_b64", "image"):
            val = ak.get(key)
            if isinstance(val, str) and val:
                return val, ak.get("image_mime") or "image/png", None

    return None, None, None

def _json_or_none(s: Optional[str]) -> Optional[Union[dict, list]]:
    if not s:
        return None
    try:
        val = json.loads(s)
        if isinstance(val, (dict, list)):
            return val
    except Exception:
        pass
    return None

# ---- Retry helpers (overload-aware) ----
def _is_overload_error(err: Exception) -> bool:
    """
    Detect 'overloaded' / retryable capacity errors from vendors.
    - Anthropic returns HTTP 529 'Overloaded'.
    - Some SDKs wrap the response; we check common attributes + message text.
    """
    msg = (str(err) or "").lower()
    if "overloaded" in msg or "status 529" in msg or "code: 529" in msg:
        return True
    status = getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)
    if status == 529:
        return True
    return False

def _backoff_delay(attempt: int, base: float = 0.6, max_backoff: float = 6.0) -> float:
    """
    Exponential backoff with jitter:
    attempt=0 -> ~base .. 2*base
    attempt=1 -> ~2*base .. 4*base
    """
    low = base * (2 ** attempt)
    high = base * (2 ** (attempt + 1))
    return min(random.uniform(low, high), max_backoff)

async def _ainvoke_with_retries(llm, msgs, *, max_retries: int = 3, base_delay: float = 0.6):
    """
    Invoke the model with limited retries on 'overloaded' errors.
    Raises HTTPException with 503 on persistent overload, 502 on other errors.
    """
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await llm.ainvoke(msgs)
        except Exception as e:
            if _is_overload_error(e) and attempt < max_retries:
                delay = _backoff_delay(attempt, base=base_delay)
                await asyncio.sleep(delay)
                continue
            last_err = e
            break

    if last_err and _is_overload_error(last_err):
        raise HTTPException(status_code=503, detail="Provider is overloaded. Please retry shortly.")
    if last_err:
        raise HTTPException(status_code=502, detail=str(last_err))
    raise HTTPException(status_code=500, detail="Unknown error during model invocation")

# =============================================================================
# JSON endpoints for the frontend ("analyze")
# =============================================================================

@router.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(None),
    image: UploadFile = File(None),                 # backward compat (ignored if 'file' is present)
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),              # informational only
    provider: Optional[str] = Form(None),           # informational only
    wire: Optional[str] = Form(None),               # preferred: "provider:model"
    system: Optional[str] = Form(None),
    model_params: Optional[str] = Form(None),       # JSON string
):
    up = _choose_upload(image=image, file_=file)
    if not up.content_type or not _is_png_or_jpeg(up.content_type):
        raise HTTPException(400, "Only image/png and image/jpeg are supported")
    raw = await up.read()
    if not raw:
        raise HTTPException(400, "Empty image upload")

    if not wire or ":" not in wire:
        raise HTTPException(400, "Missing 'wire' (format: 'provider:model')")

    reg = _registry(request)
    pkey, model_name, ptype = _provider_type_from_wire(reg, wire)

    # Build messages
    msgs: List[BaseMessage] = []
    msgs.extend(_system_msg_if(system))
    msgs.append(_build_vision_human_message(ptype, prompt, up.content_type, raw))

    # Params
    kwargs = {}
    if model_params:
        try:
            kwargs = json.loads(model_params)
            if not isinstance(kwargs, dict):
                raise ValueError
        except Exception:
            raise HTTPException(400, "model_params must be valid JSON object")

    # Build a fresh model instance with params and invoke (with retries)
    llm = resolve_and_init_from_registry(registry=reg, wire=wire, params=kwargs)
    out = await _ainvoke_with_retries(llm, msgs, max_retries=3, base_delay=0.6)

    text = _extract_text_from_output(out)
    img_b64, img_mime, img_url = _extract_image_from_output(out)

    return {
        "text": text or None,
        "json": None,
        "image_base64": img_b64,
        "image_mime": img_mime,
        "image_url": img_url,
        "model": model_name,
        "provider": pkey,
    }

# =============================================================================
# SSE endpoints (kept for parity with your LangGraph chat)
# =============================================================================

@router.post("/single/stream")
async def vision_single_stream(
    request: Request,
    wire: str = Form(...),                       # "provider:model"
    image: UploadFile = File(None),              # jpeg/png (legacy name)
    file: UploadFile = File(None),               # jpeg/png (preferred)
    prompt: Optional[str] = Form(None),          # optional user text
    model_params: Optional[str] = Form(None),    # JSON string (per-model params)
    system: Optional[str] = Form(None),          # optional system prompt
    thread_id: Optional[str] = Form("vision"),
):
    up = _choose_upload(image=image, file_=file)
    if not up.content_type or not _is_png_or_jpeg(up.content_type):
        raise HTTPException(400, "Only image/png and image/jpeg are supported")
    raw = await up.read()
    if not raw:
        raise HTTPException(400, "Empty image upload")

    reg = _registry(request)
    pkey, model_name, ptype = _provider_type_from_wire(reg, wire)

    # Build messages
    msgs: List[BaseMessage] = []
    msgs.extend(_system_msg_if(system))
    msgs.append(_build_vision_human_message(ptype, prompt, up.content_type, raw))

    # Build graph with params
    params_dict: Dict[str, Any] = {}
    if model_params:
        try:
            params_dict = json.loads(model_params)
            if not isinstance(params_dict, dict):
                raise ValueError
        except Exception:
            raise HTTPException(400, "model_params must be valid JSON object")

    compiled, _ = build_single_model_graph(
        registry=reg,
        wire=wire,
        model_kwargs=params_dict,
        memory_backend=_memory_backend(request),
    )

    async def gen():
        yield _sse_event({"type": "open", "scope": "single"}, event="open")
        await asyncio.sleep(0)
        cfg = {"configurable": {"thread_id": thread_id}}

        last_beat = asyncio.get_event_loop().time()
        got_any = False

        async for ev in compiled.astream_events({"messages": msgs, "meta": {}}, config=cfg):
            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}

            now = asyncio.get_event_loop().time()
            if now - last_beat > 10:
                yield _sse_comment("hb"); last_beat = now

            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = ""
                c = getattr(chunk, "content", None)
                if isinstance(c, str) and c:
                    piece = c
                elif isinstance(getattr(chunk, "text", None), str):
                    piece = chunk.text
                if piece:
                    got_any = True
                    yield _sse_event({"type": "delta", "scope": "single", "delta": piece, "done": False})

            elif et == "on_chat_model_end":
                if not got_any:
                    out = data.get("output")
                    final_txt = ""
                    if out is not None:
                        if isinstance(getattr(out, "content", None), str):
                            final_txt = out.content
                        elif isinstance(getattr(out, "text", None), str):
                            final_txt = out.text
                    if final_txt:
                        yield _sse_event({"type": "delta", "scope": "single", "delta": final_txt, "done": False})

            elif et == "on_chat_model_error":
                err = data.get("error")
                yield _sse_event({"type": "error", "scope": "single", "error": str(err) if err else "Unknown error"}, event="error")

        yield _sse_event({"type": "done", "scope": "single", "done": True}, event="done")

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)

@router.post("/multi/stream")
async def vision_multi_stream(
    request: Request,
    targets: str = Form(...),                   # JSON: ["prov:model", ...]
    image: UploadFile = File(None),
    file: UploadFile = File(None),
    prompt: Optional[str] = Form(None),
    per_model_params: Optional[str] = Form(None),  # JSON: { "prov:model": {...}, ... }
    system: Optional[str] = Form(None),
    thread_id: Optional[str] = Form("vision-compare"),
):
    up = _choose_upload(image=image, file_=file)
    if not up.content_type or not _is_png_or_jpeg(up.content_type):
        raise HTTPException(400, "Only image/png and image/jpeg are supported")
    raw = await up.read()
    if not raw:
        raise HTTPException(400, "Empty image upload")

    try:
        target_list = json.loads(targets)
        if not isinstance(target_list, list) or not all(isinstance(x, str) and ":" in x for x in target_list):
            raise ValueError
    except Exception:
        raise HTTPException(400, "targets must be JSON array of 'provider:model' strings")

    pmap: Dict[str, str] = {}  # wire -> provider_type
    reg = _registry(request)
    for w in target_list:
        pkey, _ = w.split(":", 1)
        pmap[w] = (reg.provider_type(pkey) or "").lower()

    def msgs_for_wire(w: str) -> List[BaseMessage]:
        content_msg = _build_vision_human_message(pmap[w], prompt, up.content_type, raw)
        msgs = _system_msg_if(system) + [content_msg]
        return msgs

    # Parse params
    per_params: Dict[str, Dict[str, Any]] = {}
    if per_model_params:
        try:
            per_params = json.loads(per_model_params)
            if not isinstance(per_params, dict):
                raise ValueError
        except Exception:
            raise HTTPException(400, "per_model_params must be a JSON object")

    compiled, _ = build_multi_model_graph(
        registry=reg,
        wires=target_list,
        per_model_params=per_params,
        memory_backend=_memory_backend(request),
    )
    node_to_wire = getattr(compiled, "_node_to_wire", {}) or {}

    async def gen():
        yield _sse_event({"type": "open", "scope": "multi", "models": target_list}, event="open")
        await asyncio.sleep(0)
        cfg = {"configurable": {"thread_id": thread_id}}

        init_state = {
            "messages": [],         # will be set below
            "targets": target_list,
            "results": {},
            "errors": {},
        }

        last_beat = asyncio.get_event_loop().time()
        streamed_any: Dict[str, bool] = {w: False for w in target_list}

        # Use identical messages for all nodes (prompt + image). If you need per-wire variation,
        # adapt your graph to accept per-node messages and call msgs_for_wire(w) at node execution.
        first_wire = target_list[0]
        init_state["messages"] = msgs_for_wire(first_wire)

        async for ev in compiled.astream_events(init_state, config=cfg):
            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}
            node = meta.get("langgraph_node")

            now = asyncio.get_event_loop().time()
            if now - last_beat > 10:
                yield _sse_comment("hb"); last_beat = now

            wire = (meta.get("wire") or meta.get("model") or "")
            if not wire and node and node in node_to_wire:
                wire = node_to_wire[node]

            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = ""
                c = getattr(chunk, "content", None)
                if isinstance(c, str) and c:
                    piece = c
                elif isinstance(getattr(chunk, "text", None), str):
                    piece = chunk.text
                if piece and wire:
                    streamed_any[wire] = True
                    yield _sse_event({"type": "delta", "scope": "multi", "model": wire, "node": node, "text": piece, "done": False})

            elif et == "on_chat_model_end":
                if not streamed_any.get(wire, False):
                    out = data.get("output")
                    final_txt = ""
                    if out is not None:
                        if isinstance(getattr(out, "content", None), str):
                            final_txt = out.content
                        elif isinstance(getattr(out, "text", None), str):
                            final_txt = out.text
                    if final_txt and wire:
                        yield _sse_event({"type": "delta", "scope": "multi", "model": wire, "node": node, "text": final_txt, "done": False})

                if wire:
                    yield _sse_event({"type": "done", "scope": "multi", "model": wire, "done": True}, event="done")

            elif et == "on_chat_model_error":
                err = data.get("error")
                if wire:
                    yield _sse_event({"type": "error", "scope": "multi", "model": wire, "node": node, "error": str(err) if err else "Unknown error", "done": False}, event="error")

        # Ensure all models are marked done
        for w in target_list:
            yield _sse_event({"type": "done", "scope": "multi", "model": w, "done": True}, event="done")

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)
