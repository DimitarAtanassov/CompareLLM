# routers/langgraph.py
from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from graphs.factory import build_single_model_graph, build_multi_model_graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

router = APIRouter(prefix="/langgraph", tags=["langgraph"])

# ---------------------------
# Ingress message sanitizer
# ---------------------------
def _lc_messages_in(msgs: list[dict | BaseMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in msgs or []:
        if isinstance(m, BaseMessage):
            # Coerce non-string content defensively
            c = m.content if isinstance(m.content, (str, list)) else str(m.content)
            m.content = c  # type: ignore[attr-defined]
            out.append(m)
            continue

        role = (m or {}).get("role")
        content = (m or {}).get("content", "")
        if not isinstance(content, (str, list)):
            content = str(content)

        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("user", "human"):
            out.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out

# ---------------------------
# SSE helpers
# ---------------------------
def _sse_event(data: Dict[str, Any], event: Optional[str] = None) -> bytes:
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    lines.append("")  # blank line terminator
    return ("\n".join(lines) + "\n").encode("utf-8")

def _sse_comment(comment: str = "") -> bytes:
    return (f":{comment}\n\n").encode("utf-8")

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# ---------------------------
# Text extraction (robust)
# ---------------------------
def _extract_from_piece(piece: Any) -> str:
    """Handle LC chunk pieces that might be dicts like {'type':'text','text':'...'}."""
    if isinstance(piece, str):
        return piece
    if isinstance(piece, dict):
        # common LC shapes
        t = piece.get("text")
        if isinstance(t, str):
            return t
        # anthropic/gemini style
        if piece.get("type") == "text" and isinstance(piece.get("text"), str):
            return piece["text"]
    return ""

def _chunk_text(chunk: Any) -> str:
    """
    Try hard to find text in LangChain message chunks across providers.
    Order:
      1) chunk.content (str)
      2) chunk.text (str)
      3) chunk.content (list[ {text: \"...\"} ])
      4) chunk.delta (str | list | dict)
    """
    # direct string content
    c = getattr(chunk, "content", None)
    if isinstance(c, str) and c:
        return c

    # direct text
    t = getattr(chunk, "text", None)
    if isinstance(t, str) and t:
        return t

    # list content with text parts
    if isinstance(c, list):
        parts: List[str] = []
        for p in c:
            s = _extract_from_piece(p)
            if s:
                parts.append(s)
        if parts:
            return "".join(parts)

    # delta-based providers
    d = getattr(chunk, "delta", None)
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        s = d.get("content") or d.get("text")
        if isinstance(s, str):
            return s
        if isinstance(s, list):
            joined = "".join(_extract_from_piece(p) for p in s)
            if joined:
                return joined
    if isinstance(d, list):
        joined = "".join(_extract_from_piece(p) for p in d)
        if joined:
            return joined

    return ""

def _end_text_from_event_data(data: Dict[str, Any]) -> str:
    """
    When we reach on_chat_model_end, try to fetch the final text:
      - data.get(\"output\") may be an AIMessage with .content/.text
      - or data.get(\"generations\")[0][0].text in some LC versions
    """
    out = data.get("output")
    if out is not None:
        # try like a message
        txt = getattr(out, "content", None)
        if isinstance(txt, str) and txt:
            return txt
        txt = getattr(out, "text", None)
        if isinstance(txt, str) and txt:
            return txt
        # list content parts
        if isinstance(getattr(out, "content", None), list):
            parts = []
            for p in getattr(out, "content"):
                parts.append(_extract_from_piece(p))
            return "".join(parts)

    gens = data.get("generations")
    # generations often like [[GenerationChunk(text=...)]] or list[list[...]]
    if isinstance(gens, list) and gens:
        g0 = gens[0]
        # either a list or a single generation
        if isinstance(g0, list) and g0:
            g0 = g0[0]
        txt = getattr(g0, "text", None)
        if isinstance(txt, str) and txt:
            return txt
        c = getattr(g0, "message", None) or getattr(g0, "generation_info", None)
        if c:
            v = getattr(c, "text", None) or getattr(c, "content", None)
            if isinstance(v, str) and v:
                return v
    return ""

# ---------------------------
# Routes
# ---------------------------
@router.post("/chat/single/stream")
async def chat_single_stream(req: Request):
    body = await req.json()
    wire: str = body.get("wire")
    if not wire:
        raise HTTPException(400, "Missing 'wire'")
    messages: List[Dict[str, str]] = body.get("messages", [])
    model_params: Dict[str, Any] = body.get("model_params") or {}
    thread_id: Optional[str] = body.get("thread_id") or "default"

    # registry-aware builder
    registry = getattr(req.app.state, "registry", None)
    if registry is None:
        raise HTTPException(500, "Model registry is not initialized")

    # ✅ Use the shared memory saver from app state
    memory_backend = getattr(req.app.state, "graph_memory", None)

    graph, _ = build_single_model_graph(
        registry,
        wire,
        model_kwargs=model_params,
        memory_backend=memory_backend,  # 👈 pass shared saver
    )

    async def gen():
        # OPEN
        yield _sse_event({"type": "open", "scope": "single"}, event="open")
        await asyncio.sleep(0)

        config = {"configurable": {"thread_id": thread_id}}
        last_beat = asyncio.get_event_loop().time()
        emitted_any = False

        # ✅ sanitize incoming messages
        sanitized = _lc_messages_in(messages)

        async for ev in graph.astream_events({"messages": sanitized, "meta": {}}, config=config):
            # heartbeat ~10s
            now = asyncio.get_event_loop().time()
            if now - last_beat > 10:
                yield _sse_comment("hb")
                last_beat = now

            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}

            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = _chunk_text(chunk) if chunk is not None else ""
                if piece:
                    emitted_any = True
                    yield _sse_event({
                        "type": "delta",
                        "scope": "single",
                        "node": meta.get("langgraph_node"),
                        "delta": piece,
                        "done": False,
                    })

            elif et == "on_chat_model_end":
                # Fallback if nothing streamed: try to emit final text
                if not emitted_any:
                    final_text = _end_text_from_event_data(data)
                    if final_text:
                        yield _sse_event({
                            "type": "delta",
                            "scope": "single",
                            "node": meta.get("langgraph_node"),
                            "delta": final_text,
                            "done": False,
                        })

            elif et == "on_chat_model_error":
                err = data.get("error")
                yield _sse_event({
                    "type": "error",
                    "scope": "single",
                    "node": meta.get("langgraph_node"),
                    "error": str(err) if err else "Unknown error",
                    "done": False,
                }, event="error")

        # DONE
        yield _sse_event({"type": "done", "scope": "single", "done": True}, event="done")

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

    registry = getattr(req.app.state, "registry", None)
    if registry is None:
        raise HTTPException(500, "Model registry is not initialized")

    # Use the shared memory saver from app state
    memory_backend = getattr(req.app.state, "graph_memory", None)

    graph, _ = build_multi_model_graph(
        registry,
        targets,
        per_model_params=per_model_params,
        memory_backend=memory_backend,  #  pass shared saver
    )
    node_to_wire = getattr(graph, "_node_to_wire", {}) or {}

    async def gen():
        # OPEN
        yield _sse_event({"type": "open", "scope": "multi", "models": targets}, event="open")
        await asyncio.sleep(0)

        config = {"configurable": {"thread_id": thread_id}}

        # ✅ sanitize incoming messages
        sanitized = _lc_messages_in(messages)

        init_state = {
            "messages": sanitized,
            "targets": targets,
            "results": {},
            "errors": {},
        }

        last_beat = asyncio.get_event_loop().time()
        emitted_any: Dict[str, bool] = {w: False for w in targets}

        async for ev in graph.astream_events(init_state, config=config):
            now = asyncio.get_event_loop().time()
            if now - last_beat > 10:
                yield _sse_comment("hb")
                last_beat = now

            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}
            node = meta.get("langgraph_node")

            # Try to get a proper wire id for the frontend
            wire = (meta.get("wire") or meta.get("model") or "")  # often empty
            if not wire and node and node in node_to_wire:
                wire = node_to_wire[node]

            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = _chunk_text(chunk) if chunk is not None else ""
                if piece and wire:
                    emitted_any[wire] = True
                    yield _sse_event({
                        "type": "delta",
                        "scope": "multi",
                        "model": wire,
                        "node": node,
                        "text": piece,
                        "done": False
                    })

            elif et == "on_chat_model_end":
                if wire:
                    # Fallback if nothing streamed for this model
                    if not emitted_any.get(wire, False):
                        final_text = _end_text_from_event_data(data)
                        if final_text:
                            yield _sse_event({
                                "type": "delta",
                                "scope": "multi",
                                "model": wire,
                                "node": node,
                                "text": final_text,
                                "done": False
                            })

                    # ALWAYS mark this model as done as soon as its node ends
                    yield _sse_event({
                        "type": "done",
                        "scope": "multi",
                        "model": wire,
                        "done": True
                    }, event="done")

            elif et == "on_chat_model_error":
                if wire:
                    err = data.get("error")
                    yield _sse_event({
                        "type": "error",
                        "scope": "multi",
                        "model": wire,
                        "node": node,
                        "error": str(err) if err else "Unknown error",
                        "done": False
                    }, event="error")

        # Safety net: Ensure all requested models have a done marker
        for w in targets:
            yield _sse_event({"type": "done", "scope": "multi", "model": w, "done": True}, event="done")

    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)

