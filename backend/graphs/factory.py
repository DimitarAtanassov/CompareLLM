from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import SingleState, MultiState, EmbeddingState
from core.model_factory import resolve_and_init_from_registry

# =========================
# MULTI-MODEL EMBEDDING SEARCH GRAPH (non-streaming)
# =========================
def build_embedding_comparison_graph(
    registry: Any,  # EmbeddingRegistry
    embedding_keys: List[str],
    dataset_id: str,
    memory_backend: Optional[Any] = None,
):
    from langchain_core.vectorstores import VectorStore
    from core.embedding_registry import EmbeddingRegistry as EmbeddingRegistryType
    import re

    emb_reg: EmbeddingRegistryType = registry
    g = StateGraph(EmbeddingState)

    def _safe_node_name(w: str, idx: int) -> str:
        return f"e_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', w)}"

    key_to_node = {k: _safe_node_name(k, i) for i, k in enumerate(embedding_keys)}

    def _make_search_node(emb_key: str):
        async def run_search(state: EmbeddingState):
            query = state["query"]
            search_params = state["search_params"]
            k = search_params.get("k", 5)
            stype = (search_params.get("search_type") or "similarity").strip().lower()
            store_id = f"{dataset_id}::{emb_key}"

            try:
                vs: VectorStore = emb_reg.get_store(store_id)
                docs_with_scores = []

                if stype == "similarity":
                    if search_params.get("with_scores"):
                        results = await vs.asimilarity_search_with_score(query=query, k=k)
                        docs_with_scores = [{"doc": doc, "score": score} for doc, score in results]
                    else:
                        results = await vs.asimilarity_search(query=query, k=k)
                        docs_with_scores = [{"doc": doc, "score": None} for doc in results]
                else:
                    search_kwargs: Dict[str, Any] = {"k": k}
                    if stype == "mmr":
                        if "fetch_k" in search_params:
                            search_kwargs["fetch_k"] = search_params["fetch_k"]
                        if "lambda_mult" in search_params:
                            search_kwargs["lambda_mult"] = search_params["lambda_mult"]
                    elif stype == "similarity_score_threshold":
                        if "score_threshold" in search_params:
                            search_kwargs["score_threshold"] = search_params["score_threshold"]
                    else:
                        raise ValueError(f"Unsupported search type: {stype}")

                    retriever = vs.as_retriever(search_type=stype, search_kwargs=search_kwargs)
                    retrieved_docs = await retriever.ainvoke(query)
                    
                    if search_params.get("with_scores"):
                        qvec = vs.embeddings.embed_query(query)
                        scored = await vs.asimilarity_search_with_score_by_vector(qvec, k=max(k, len(retrieved_docs)))
                        score_by_id: Dict[Optional[str], float] = {getattr(d, "id", None): float(s) for d, s in scored}
                        docs_with_scores = [{"doc": d, "score": score_by_id.get(getattr(d, "id", None))} for d in retrieved_docs]
                    else:
                        docs_with_scores = [{"doc": doc, "score": None} for doc in retrieved_docs]

                items = [{
                    "page_content": dws["doc"].page_content,
                    "metadata": dws["doc"].metadata,
                    "score": dws["score"]
                } for dws in docs_with_scores]

                return {"results": {emb_key: {"items": items[:k]}}}

            except Exception as e:
                return {"errors": {emb_key: f"{type(e).__name__}: {e}"}}
        return run_search

    for emb_key, node_name in key_to_node.items():
        g.add_node(node_name, _make_search_node(emb_key))

    def router(state: EmbeddingState):
        return [key_to_node[k] for k in state["targets"] if k in key_to_node]

    g.add_conditional_edges(START, router)

    for node_name in key_to_node.values():
        g.add_edge(node_name, END)

    checkpointer = memory_backend or InMemorySaver()
    compiled = g.compile(checkpointer=checkpointer)
    return compiled, checkpointer


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

    # ---- Helper to create a model-invoking node ----
    def _make_chatbot_node(wire: str):
        async def chatbot_node(state: MultiState):
            # Resolve the specific LLM from the registry
            model_kwargs = per_model_params.get(wire, {})
            llm = resolve_and_init_from_registry(registry, wire, model_kwargs)

            # Stream back deltas
            got_any = False
            async for chunk in llm.astream(_lc_messages(state["messages"])):
                piece = _chunk_text(chunk)
                if piece:
                    got_any = True
                    yield {"results": {wire: piece}}
            
            # One-shot fallback
            if not got_any:
                resp = await llm.ainvoke(_lc_messages(state["messages"]))
                txt = _to_safe_text(resp)
                if txt:
                    yield {"results": {wire: txt}}

        return chatbot_node

    # ---- Graph construction ----
    def _safe_node_name(w: str, idx: int) -> str:
        import re
        return f"w_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', w)}"

    wire_to_node = {w: _safe_node_name(w, i) for i, w in enumerate(wires)}

    # Add a node for each model wire
    for wire, node_name in wire_to_node.items():
        g.add_node(node_name, _make_chatbot_node(wire))

    # Router: fan out to all targeted models
    def router(state: MultiState):
        return [wire_to_node[t] for t in state["targets"] if t in wire_to_node]

    g.add_conditional_edges(START, router)

    # All model nodes connect to the end
    for node_name in wire_to_node.values():
        g.add_edge(node_name, END)

    checkpointer = memory_backend or InMemorySaver()
    compiled = g.compile(checkpointer=checkpointer)
    return compiled, checkpointer
