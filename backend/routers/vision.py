# routers/vision.py
from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse

from services.vision_service import VisionService
from graphs.factory import build_single_model_graph, build_multi_model_graph

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

STREAM_HEADERS = VisionService.STREAM_HEADERS

# Dependency injection for VisionService
def get_vision_service() -> VisionService:
    return VisionService()

# =============================================================================
# JSON endpoints for the frontend ("analyze")
# =============================================================================

@router.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(None),
    image: UploadFile = File(None),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
    wire: Optional[str] = Form(None),
    system: Optional[str] = Form(None),
    model_params: Optional[str] = Form(None),
    vision_service: VisionService = Depends(get_vision_service),
):
    up = vision_service.choose_upload(image=image, file_=file)
    vision_service.validate_upload(up)
    raw = await vision_service.read_upload(up)

    if not wire or ":" not in wire:
        raise HTTPException(400, "Missing 'wire' (format: 'provider:model')")

    reg = vision_service.get_registry(request)
    pkey, model_name, ptype = vision_service.provider_type_from_wire(reg, wire)

    msgs: List[Any] = []
    msgs.extend(vision_service._system_msg_if(system))
    msgs.append(vision_service._build_vision_human_message(ptype, prompt, up.content_type, raw))

    kwargs = {}
    if model_params:
        try:
            kwargs = json.loads(model_params)
            if not isinstance(kwargs, dict):
                raise ValueError
        except Exception:
            raise HTTPException(400, "model_params must be valid JSON object")

    # Use the service to resolve/init the model
    llm = vision_service.resolve_and_init_model(reg, wire, kwargs)
    out = await vision_service.ainvoke_with_retries(llm, msgs, max_retries=3, base_delay=0.6)

    text = vision_service._extract_text_from_output(out)
    img_b64, img_mime, img_url = vision_service._extract_image_from_output(out)

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
    wire: str = Form(...),
    image: UploadFile = File(None),
    file: UploadFile = File(None),
    prompt: Optional[str] = Form(None),
    model_params: Optional[str] = Form(None),
    system: Optional[str] = Form(None),
    thread_id: Optional[str] = Form("vision"),
    vision_service: VisionService = Depends(get_vision_service),
):
    up = vision_service.choose_upload(image=image, file_=file)
    vision_service.validate_upload(up)
    raw = await vision_service.read_upload(up)

    reg = vision_service.get_registry(request)
    pkey, model_name, ptype = vision_service.provider_type_from_wire(reg, wire)

    msgs: List[Any] = []
    msgs.extend(vision_service._system_msg_if(system))
    msgs.append(vision_service._build_vision_human_message(ptype, prompt, up.content_type, raw))

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
        memory_backend=vision_service.get_memory_backend(request),
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
    targets: str = Form(...),
    image: UploadFile = File(None),
    file: UploadFile = File(None),
    prompt: Optional[str] = Form(None),
    per_model_params: Optional[str] = Form(None),
    system: Optional[str] = Form(None),
    thread_id: Optional[str] = Form("vision-compare"),
    vision_service: VisionService = Depends(get_vision_service),
):
    up = vision_service.choose_upload(image=image, file_=file)
    vision_service.validate_upload(up)
    raw = await vision_service.read_upload(up)

    try:
        target_list = json.loads(targets)
        if not isinstance(target_list, list) or not all(isinstance(x, str) and ":" in x for x in target_list):
            raise ValueError
    except Exception:
        raise HTTPException(400, "targets must be JSON array of 'provider:model' strings")

    pmap: Dict[str, str] = {}
    reg = vision_service.get_registry(request)
    for w in target_list:
        pkey, _ = w.split(":", 1)
        pmap[w] = (reg.provider_type(pkey) or "").lower()

    def msgs_for_wire(w: str) -> List[Any]:
        content_msg = vision_service._build_vision_human_message(pmap[w], prompt, up.content_type, raw)
        msgs = vision_service._system_msg_if(system) + [content_msg]
        return msgs

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
        memory_backend=vision_service.get_memory_backend(request),
    )
    node_to_wire = getattr(compiled, "_node_to_wire", {}) or {}

    async def gen():
        yield _sse_event({"type": "open", "scope": "multi", "models": target_list}, event="open")
        await asyncio.sleep(0)
        cfg = {"configurable": {"thread_id": thread_id}}

        init_state = {
            "messages": [],
            "targets": target_list,
            "results": {},
            "errors": {},
        }

        last_beat = asyncio.get_event_loop().time()
        streamed_any: Dict[str, bool] = {w: False for w in target_list}

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

        for w in target_list:
            yield _sse_event({"type": "done", "scope": "multi", "model": w, "done": True}, event="done")

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)
