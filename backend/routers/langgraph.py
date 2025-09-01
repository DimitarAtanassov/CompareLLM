# routers/langgraph.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from graphs.factory import build_single_model_graph, build_multi_model_graph

router = APIRouter(prefix="/langgraph", tags=["langgraph"])

def _ndjson(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# Minimal chunk->text extractor (keeps your existing shapes)
def _chunk_text(chunk: Any) -> str:
    c = getattr(chunk, "content", None)
    if isinstance(c, str):
        return c
    parts = []
    if isinstance(c, list):
        for p in c:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                parts.append(t)
    t = getattr(chunk, "text", None)
    if isinstance(t, str):
        parts.append(t)
    return "".join(parts)


@router.post("/chat/single/stream")
async def chat_single_stream(req: Request):
    body = await req.json()
    wire: str = body.get("wire")
    if not wire:
        raise HTTPException(400, "Missing 'wire'")
    messages: List[Dict[str, str]] = body.get("messages", [])
    model_params: Dict[str, Any] = body.get("model_params") or {}
    thread_id: Optional[str] = body.get("thread_id") or "default"

    graph, _ = build_single_model_graph(wire, model_kwargs=model_params)

    async def gen():
        # open the SSE
        yield b":open\n\n"
        config = {"configurable": {"thread_id": thread_id}}

        # Stream LOW-LEVEL events (no tick buffering)
        async for ev in graph.astream_events({"messages": messages, "meta": {}}, config=config):
            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}

            # Token deltas as they happen
            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = _chunk_text(chunk) if chunk is not None else ""
                if piece:
                    yield _ndjson({
                        "type": "delta",
                        "scope": "single",
                        "node": meta.get("langgraph_node"),
                        "delta": piece,
                        "done": False,
                    })

            # Optional: surface model errors
            if et == "on_chat_model_error":
                err = data.get("error")
                yield _ndjson({
                    "type": "error",
                    "scope": "single",
                    "node": meta.get("langgraph_node"),
                    "error": str(err) if err else "Unknown error",
                    "done": False,
                })

        # End marker
        yield _ndjson({"type": "done", "scope": "single", "done": True})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)


@router.post("/chat/multi/stream")
async def chat_multi_stream(req: Request):
    body = await req.json()
    targets = body.get("targets", [])
    if not targets:
        raise HTTPException(400, "Missing 'targets'[]")
    messages = body.get("messages", [])
    per_model_params = body.get("per_model_params") or {}
    thread_id = body.get("thread_id") or "compare"

    graph, _ = build_multi_model_graph(targets, per_model_params=per_model_params)

    async def gen():
        yield b":open\n\n"
        config = {"configurable": {"thread_id": thread_id}}
        init_state = {"messages": messages, "targets": targets, "results": {}, "per_model_params": per_model_params}

        async for ev in graph.astream_events(init_state, config=config):
            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}
            wire = (meta.get("wire") or meta.get("model") or "")  # set by llm.with_config(metadata={"wire": ...})
            node = meta.get("langgraph_node")

            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = _chunk_text(chunk) if chunk is not None else ""
                if piece and wire:
                    # send FULL-SO-FAR or PIECES — your UI overwrites, so full-so-far isn’t required.
                    # Here we just send PIECES to keep payloads tiny:
                    yield _ndjson({
                        "type": "delta",
                        "scope": "multi",
                        "model": wire,
                        "node": node,
                        "text": piece,   # piece-wise
                        "done": False
                    })

            elif et == "on_chat_model_end":
                if wire:
                    yield _ndjson({"type": "done", "scope": "multi", "model": wire, "done": True})

            elif et == "on_chat_model_error":
                err = data.get("error")
                if wire:
                    yield _ndjson({
                        "type": "error",
                        "scope": "multi",
                        "model": wire,
                        "node": node,
                        "error": str(err) if err else "Unknown error",
                        "done": False
                    })

        # Ensure all requested models have a done marker
        for w in targets:
            yield _ndjson({"type": "done", "scope": "multi", "model": w, "done": True})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)
