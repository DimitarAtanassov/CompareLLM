from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from graphs.state import SingleState, MultiState, EmbeddingState
from core.model_factory import resolve_and_init_from_registry
from core.embedding_registry import EmbeddingRegistry


def _log(msg: str) -> None:
    print(f"[GraphService] {msg}")


class GraphService:
    """
    Service layer for building LangGraph state graphs.
    Handles graph construction for single-model, multi-model, and embedding comparison scenarios.
    """
    
    def __init__(self) -> None:
        _log("Initialized")
    
    # ---------- Message Conversion ----------
    def convert_to_langchain_messages(self, msgs: List[Any]) -> List[BaseMessage]:
        """Convert generic message format to LangChain messages."""
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
    
    # ---------- Text Extraction ----------
    def extract_text_from_chunk(self, chunk: Any) -> str:
        """Extract text from various LangChain chunk formats."""
        # Direct string content
        c = getattr(chunk, "content", None)
        # Handle callable content (older/newer LC variants)
        if callable(c):
            try:
                c = c()
            except Exception:
                c = None
        if isinstance(c, str) and c:
            return c
        
        # Direct .text
        t = getattr(chunk, "text", None)
        if callable(t):
            try:
                t = t()
            except Exception:
                t = None
        if isinstance(t, str) and t:
            return t
        
        # List content
        if isinstance(c, list):
            parts: List[str] = []
            for p in c:
                s = self._extract_piece(p)
                if s:
                    parts.append(s)
            if parts:
                return "".join(parts)
        
        # Delta shapes
        d = getattr(chunk, "delta", None)
        if isinstance(d, str):
            return d
        if isinstance(d, list):
            joined = "".join(self._extract_piece(p) for p in d)
            if joined:
                return joined
        if isinstance(d, dict):
            s = d.get("content") or d.get("text")
            if isinstance(s, str):
                return s
            if isinstance(s, list):
                joined = "".join(self._extract_piece(p) for p in s)
                if joined:
                    return joined
        return ""
    
    def _extract_piece(self, part: Any) -> str:
        """Extract text from message part."""
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            t = part.get("text")
            if isinstance(t, str):
                return t
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                return part["text"]
        return ""
    
    def to_safe_text(self, obj: Any) -> str:
        """Ultimate fallback: turn anything that looks like a model response into text."""
        # Try robust chunk reader
        s = self.extract_text_from_chunk(obj)
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
                    joined = "".join(self._extract_piece(p) for p in val)
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
                joined = "".join(self._extract_piece(p) for p in c)
                if joined.strip():
                    return joined
        
        # Last resort
        try:
            return str(obj)
        except Exception:
            return ""
    
    def _safe_node_name(self, identifier: str, idx: int) -> str:
        """Create a safe node name for LangGraph."""
        return f"n_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', identifier)}"
    
    # ---------- Single Model Graph ----------
    def build_single_model_graph(
        self,
        registry: Any,
        wire: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        memory_backend: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Build a graph for single model chat."""
        model_kwargs = model_kwargs or {}
        llm = resolve_and_init_from_registry(registry, wire, model_kwargs)
        
        g = StateGraph(SingleState)
        
        async def chatbot(state: SingleState):
            got_any = False
            async for chunk in llm.astream(self.convert_to_langchain_messages(state["messages"])):
                piece = self.extract_text_from_chunk(chunk)
                if piece:
                    got_any = True
                    # Stream deltas as strings
                    yield {"messages": [AIMessage(content=piece)]}
            
            if not got_any:
                # One-shot fallback: coerce the response safely to text
                resp = await llm.ainvoke(self.convert_to_langchain_messages(state["messages"]))
                txt = self.to_safe_text(resp)
                if txt:
                    # Ensure it's a plain string
                    if not isinstance(txt, str):
                        txt = str(txt)
                    yield {"messages": [AIMessage(content=txt)]}
        
        g.add_node("chatbot", chatbot)
        g.add_edge(START, "chatbot")
        g.add_edge("chatbot", END)
        
        checkpointer = memory_backend or InMemorySaver()
        compiled = g.compile(checkpointer=checkpointer)
        
        _log(f"Built single model graph for wire: {wire}")
        return compiled, checkpointer
    
    # ---------- Multi Model Graph ----------
    def build_multi_model_graph(
        self,
        registry: Any,
        wires: List[str],
        per_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        memory_backend: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Build a graph for multi-model comparison."""
        per_model_params = per_model_params or {}
        g = StateGraph(MultiState)
        
        # Helper to create a model-invoking node
        def _make_chatbot_node(wire: str):
            async def chatbot_node(state: MultiState):
                # Resolve the specific LLM from the registry
                model_kwargs = per_model_params.get(wire, {})
                llm = resolve_and_init_from_registry(registry, wire, model_kwargs)
                
                # Stream back deltas
                got_any = False
                async for chunk in llm.astream(self.convert_to_langchain_messages(state["messages"])):
                    piece = self.extract_text_from_chunk(chunk)
                    if piece:
                        got_any = True
                        yield {"results": {wire: piece}}
                
                # One-shot fallback
                if not got_any:
                    resp = await llm.ainvoke(self.convert_to_langchain_messages(state["messages"]))
                    txt = self.to_safe_text(resp)
                    if txt:
                        yield {"results": {wire: txt}}
            
            return chatbot_node
        
        # Graph construction
        wire_to_node = {w: self._safe_node_name(w, i) for i, w in enumerate(wires)}
        
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
        
        # Attach node<->wire mapping for router
        node_to_wire = {v: k for k, v in wire_to_node.items()}
        setattr(compiled, "_node_to_wire", node_to_wire)
        setattr(compiled, "_wire_to_node", wire_to_node)
        
        _log(f"Built multi model graph for wires: {wires}")
        return compiled, checkpointer
    
    # ---------- Embedding Comparison Graph ----------  
    def build_embedding_comparison_graph(
        self,
        registry: EmbeddingRegistry,
        embedding_keys: List[str],
        dataset_id: str,
        memory_backend: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Build a graph for multi-model embedding search comparison."""
        g = StateGraph(EmbeddingState)
        
        key_to_node = {k: self._safe_node_name(k, i) for i, k in enumerate(embedding_keys)}
        
        def _make_search_node(emb_key: str):
            async def run_search(state: EmbeddingState):
                query = state["query"]
                search_params = state["search_params"]
                k = search_params.get("k", 5)
                stype = (search_params.get("search_type") or "similarity").strip().lower()
                store_id = f"{dataset_id}::{emb_key}"
                
                try:
                    vs: VectorStore = registry.get_store(store_id)
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
                            scored = await vs.asimilarity_search_with_score_by_vector(
                                qvec, k=max(k, len(retrieved_docs))
                            )
                            score_by_id: Dict[Optional[str], float] = {
                                getattr(d, "id", None): float(s) for d, s in scored
                            }
                            docs_with_scores = [{
                                "doc": d, 
                                "score": score_by_id.get(getattr(d, "id", None))
                            } for d in retrieved_docs]
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
        
        _log(f"Built embedding comparison graph for keys: {embedding_keys}, dataset: {dataset_id}")
        return compiled, checkpointer
