# graphs/factory.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import SingleState, MultiState
from core.model_factory import resolve_and_init_from_registry

# =========================
# Utilities
# =========================
def _lc_messages(msgs: List[Any]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in msgs or []:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        r = (m or {}).get("role")
        c = (m or {}).get("content", "")
        if r == "system":
            out.append(SystemMessage(content=c))
        elif r == "user":
            out.append(HumanMessage(content=c))
        elif r == "assistant":
            out.append(AIMessage(content=c))
        else:
            out.append(AIMessage(content=c))
    return out


def _extract_piece(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        t = part.get("text")
        if isinstance(t, str):
            return t
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            return part["text"]
    return ""


def _chunk_text(chunk: Any) -> str:
    # 1) direct string content
    c = getattr(chunk, "content", None)
    # Handle the case where .content is a *callable* (older/newer LC variants or wrappers)
    if callable(c):
        try:
            c = c()
        except Exception:
            c = None
    if isinstance(c, str) and c:
        return c
    # 2) direct .text
    t = getattr(chunk, "text", None)
    if callable(t):
        try:
            t = t()
        except Exception:
            t = None
    if isinstance(t, str) and t:
        return t
    # 3) list content
    if isinstance(c, list):
        parts: List[str] = []
        for p in c:
            s = _extract_piece(p)
            if s:
                parts.append(s)
        if parts:
            return "".join(parts)
    # 4) delta shapes
    d = getattr(chunk, "delta", None)
    if isinstance(d, str):
        return d
    if isinstance(d, list):
        joined = "".join(_extract_piece(p) for p in d)
        if joined:
            return joined
    if isinstance(d, dict):
        s = d.get("content") or d.get("text")
        if isinstance(s, str):
            return s
        if isinstance(s, list):
            joined = "".join(_extract_piece(p) for p in s)
            if joined:
                return joined
    return ""


def _to_safe_text(obj: Any) -> str:
    """
    Ultimate fallback: turn *anything* that looks like a model response into text.
    Tries _chunk_text first, then .content/.text (calling if bound method),
    then str(obj).
    """
    # Try robust chunk reader
    s = _chunk_text(obj)
    if isinstance(s, str) and s.strip():
        return s

    # Try .content / .text directly (handling callables)
    for attr in ("content", "text"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    val = None
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list):
                joined = "".join(_extract_piece(p) for p in val)
                if joined.strip():
                    return joined

    # BaseMessage with weird content types
    if isinstance(obj, BaseMessage):
        c = obj.content
        if callable(c):
            try:
                c = c()
            except Exception:
                c = None
        if isinstance(c, str) and c.strip():
            return c
        if isinstance(c, list):
            joined = "".join(_extract_piece(p) for p in c)
            if joined.strip():
                return joined

    # Last resort
    try:
        return str(obj)
    except Exception:
        return ""


# =========================
# SINGLE-MODEL GRAPH (streaming)
# =========================
def build_single_model_graph(
    registry: Any,
    wire: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    memory_backend: Optional[Any] = None,
):
    model_kwargs = model_kwargs or {}
    llm = resolve_and_init_from_registry(registry, wire, model_kwargs)

    g = StateGraph(SingleState)

    async def chatbot(state: SingleState):
        got_any = False
        async for chunk in llm.astream(_lc_messages(state["messages"])):
            piece = _chunk_text(chunk)
            if piece:
                got_any = True
                # Stream deltas as strings
                yield {"messages": [AIMessage(content=piece)]}

        if not got_any:
            # One-shot fallback: coerce the response safely to text
            resp = await llm.ainvoke(_lc_messages(state["messages"]))
            txt = _to_safe_text(resp)
            if txt:
                # Ensure it's a plain string (AIMessage content supports list[str|dict] too,
                # but we stick to str to avoid validation edge cases)
                if not isinstance(txt, str):
                    txt = str(txt)
                yield {"messages": [AIMessage(content=txt)]}

    g.add_node("chatbot", chatbot)
    g.add_edge(START, "chatbot")
    g.add_edge("chatbot", END)

    checkpointer = memory_backend or InMemorySaver()
    compiled = g.compile(checkpointer=checkpointer)
    return compiled, checkpointer


# =========================
# MULTI-MODEL COMPARE GRAPH (streaming deltas)
# =========================
def build_multi_model_graph(
    registry: Any,
    wires: List[str],
    per_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    memory_backend: Optional[Any] = None,
):
    per_model_params = per_model_params or {}
    g = StateGraph(MultiState)

    def _safe_node_name(w: str, idx: int) -> str:
        import re
        return f"m_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', w)}"

    # Map wires <-> node names
    wire_to_node = {w: _safe_node_name(w, i) for i, w in enumerate(wires)}
    node_to_wire = {node: wire for wire, node in wire_to_node.items()}

    def _make_model_node(wire: str):
        async def run_model(state: MultiState):
            if wire not in state["targets"]:
                return
            params = per_model_params.get(wire, {})
            try:
                llm = resolve_and_init_from_registry(registry, wire, params)

                sofar = ""
                got_any = False
                async for chunk in llm.astream(_lc_messages(state["messages"])):
                    piece = _chunk_text(chunk)
                    if not piece:
                        continue
                    got_any = True
                    sofar += piece
                    yield {"results": {wire: sofar}}

                if not got_any:
                    resp = await llm.ainvoke(_lc_messages(state["messages"]))
                    final_text = _to_safe_text(resp)
                    if final_text:
                        yield {"results": {wire: final_text}}

            except Exception as e:
                yield {"errors": {wire: f"{type(e).__name__}: {e}"}}

        return run_model

    for w, node_name in wire_to_node.items():
        g.add_node(node_name, _make_model_node(w))

    def router(state: MultiState):
        return {}

    g.add_node("router", router)

    def route_to_targets(state: MultiState):
        return [wire_to_node[w] for w in state["targets"] if w in wire_to_node]

    g.add_conditional_edges("router", route_to_targets)

    for node_name in wire_to_node.values():
        g.add_edge(node_name, END)

    g.add_edge(START, "router")

    checkpointer = memory_backend or InMemorySaver()
    compiled = g.compile(checkpointer=checkpointer)

    # <-- expose node->wire so the router can recover the model id on SSE events
    setattr(compiled, "_node_to_wire", node_to_wire)

    return compiled, checkpointer
