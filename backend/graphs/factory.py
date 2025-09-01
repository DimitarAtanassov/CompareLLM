# app/backend/graphs/factory.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import SingleState, MultiState

# Cerebras integration (special-cased; init_chat_model doesn't support model_provider="cerebras")
from langchain_cerebras import ChatCerebras


# =========================
# Utilities
# =========================

def _lc_messages(msgs: List[Any]) -> List[BaseMessage]:
    """
    Accept a list of {role, content} dicts OR LangChain BaseMessage objects
    and normalize to a list[BaseMessage].
    """
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
            # default to assistant for unknown roles to avoid crashing
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
# Provider mapping
# =========================

# Only pass kwargs Cerebras supports (prevent passing unsupported args like base_url)
_CEREBRAS_ALLOWED_KWARGS = {
    "temperature",
    "max_tokens",
    "timeout",
    "max_retries",
    # add more as confirmed by the integration
}

# Your "provider:model" prefixes → LangChain provider ids
_PROVIDER_ALIASES: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "ollama": "ollama",
    "deepseek": "deepseek",
    "gemini": "google_genai",   # alias
    "google_genai": "google_genai",
    "cerebras": "cerebras",     # special-cased below
}


def _make_llm(wire: str, **kwargs):
    """
    Accepts strings like:
      - 'openai:gpt-4o'
      - 'gemini:gemini-1.5-pro-latest'
      - 'cohere:command-r'
      - 'cerebras:llama3.1-8b'
      - or raw 'gpt-4o' (no prefix)
    """
    filt = {k: v for k, v in kwargs.items() if v is not None}
    provider: Optional[str] = None
    model = wire

    if ":" in wire:
        prefix, model = wire.split(":", 1)
        provider = _PROVIDER_ALIASES.get(prefix, prefix)  # fall back to raw prefix

    # Cerebras is not supported via init_chat_model's model_provider kwarg
    if provider == "cerebras":
        cerebras_kwargs = {k: v for k, v in filt.items() if k in _CEREBRAS_ALLOWED_KWARGS}
        return ChatCerebras(model=model, **cerebras_kwargs)

    # All other providers: go through LangChain's init_chat_model
    if provider:
        return init_chat_model(model=model, model_provider=provider, **filt)

    # No explicit provider prefix: let LangChain infer from environment
    return init_chat_model(model=wire, **filt)


# =========================
# SINGLE-MODEL GRAPH (streaming)
# =========================

def build_single_model_graph(
    wire: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    memory_backend: Optional[Any] = None,
):
    """
    Build a single-model graph that streams assistant deltas via node yields.
    Your /chat/single/stream endpoint already forwards each yielded assistant delta
    as NDJSON {"type":"delta","scope":"single","delta":"<piece>"}.
    """
    model_kwargs = model_kwargs or {}
    llm = _make_llm(wire, **model_kwargs).with_config({"metadata": {"wire": wire}})

    g = StateGraph(SingleState)

    async def chatbot(state: SingleState):
        # Stream piece-by-piece; LangGraph forwards each yield as a patch
        async for chunk in llm.astream(_lc_messages(state["messages"])):
            piece = _chunk_text(chunk)
            if piece:
                # add_messages aggregator (in state.py) merges consecutive assistant parts
                yield {"messages": [{"role": "assistant", "content": piece}]}
        # If a provider didn't stream at all, there will be no yields.
        # We intentionally do not call invoke/ainvoke here: the endpoint will have already
        # propagated any yielded patches. If truly nothing streamed, the UI just won't see deltas.

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
                llm = _make_llm(wire, **params).with_config({"metadata": {"wire": wire}})

                sofar = ""
                got_any = False
                async for chunk in llm.astream(_lc_messages(state["messages"])):
                    piece = _chunk_text(chunk)
                    if not piece:
                        continue
                    got_any = True
                    sofar += piece
                    # ✅ yield on EVERY piece, but send the FULL buffer
                    yield {"results": {wire: sofar}}

                if not got_any:
                    # Fallback if the provider didn’t stream
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