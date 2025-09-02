# app/graphs/factory.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import SingleState, MultiState
from core.model_factory import resolve_and_bind_from_registry

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

def _chunk_text(chunk) -> str:
    c = getattr(chunk, "content", None)
    if isinstance(c, str):
        return c
    out = []
    if isinstance(c, list):
        for p in c:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                out.append(t)
    t = getattr(chunk, "text", None)
    if isinstance(t, str):
        out.append(t)
    return "".join(out)

# =========================
# SINGLE-MODEL GRAPH (streaming)
# =========================
def build_single_model_graph(
    registry: Any,
    wire: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    memory_backend: Optional[Any] = None,
):
    """
    Build a single-model graph that streams assistant deltas via node yields.
    """
    model_kwargs = model_kwargs or {}
    llm = resolve_and_bind_from_registry(registry, wire, model_kwargs).with_config({"metadata": {"wire": wire}})

    g = StateGraph(SingleState)

    async def chatbot(state: SingleState):
        async for chunk in llm.astream(_lc_messages(state["messages"])):
            piece = _chunk_text(chunk)
            if piece:
                yield {"messages": [{"role": "assistant", "content": piece}]}

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

    wire_to_node = {w: _safe_node_name(w, i) for i, w in enumerate(wires)}

    def _make_model_node(wire: str):
        async def run_model(state: MultiState):
            if wire not in state["targets"]:
                return
            params = per_model_params.get(wire, {})
            try:
                llm = resolve_and_bind_from_registry(registry, wire, params).with_config({"metadata": {"wire": wire}})

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
                    final_text = getattr(resp, "content", None) or getattr(resp, "text", None) or ""
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
    return g.compile(checkpointer=checkpointer), checkpointer
