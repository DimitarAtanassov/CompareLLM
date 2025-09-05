from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator
from fastapi import HTTPException
from services.graph_service import GraphService

def _log(msg: str) -> None:
    print(f"[LangGraphService] {msg}")

class LangGraphService:
    """
    Service layer for LangGraph chat operations.
    Handles graph construction, message processing, and streaming for single and multi-model chats.
    """
    
    # SSE headers for streaming responses
    STREAM_HEADERS = {
        "Cache-Control": "no-cache", 
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    
    def __init__(self) -> None:
        self.graph_service = GraphService()
        _log("Initialized")
    
    # ---------- Message Processing ----------
    def sanitize_messages(self, msgs: List[Dict[str, str] | Any]) -> List[Any]:
        """Convert raw message dicts to proper LangChain message types using GraphService."""
        # Convert dict messages to a format GraphService expects
        converted_msgs = []
        for m in msgs or []:
            if isinstance(m, dict):
                converted_msgs.append(m)
            else:
                # Assume it's already a BaseMessage
                converted_msgs.append(m)
        return self.graph_service.convert_to_langchain_messages(converted_msgs)
    
    # ---------- SSE Helpers ----------
    def sse_event(self, data: Dict[str, Any], event: Optional[str] = None) -> bytes:
        """Format data as an SSE event."""
        lines = []
        if event:
            lines.append(f"event: {event}")
        lines.append("data: " + json.dumps(data, ensure_ascii=False))
        lines.append("")  # blank line terminator
        return ("\n".join(lines) + "\n").encode("utf-8")
    
    def sse_comment(self, comment: str = "") -> bytes:
        """Format a comment for SSE heartbeat."""
        return (f":{comment}\n\n").encode("utf-8")
    
    def _extract_final_text(self, data: Dict[str, Any]) -> str:
        """Extract final text from on_chat_model_end event data."""
        out = data.get("output")
        if out is not None:
            # Try like a message
            txt = getattr(out, "content", None)
            if isinstance(txt, str) and txt:
                return txt
            txt = getattr(out, "text", None)
            if isinstance(txt, str) and txt:
                return txt
            # List content parts
            if isinstance(getattr(out, "content", None), list):
                parts = []
                for p in getattr(out, "content"):
                    parts.append(self.graph_service._extract_piece(p))
                return "".join(parts)
        
        gens = data.get("generations")
        # Generations often like [[GenerationChunk(text=...)]] or list[list[...]]
        if isinstance(gens, list) and gens:
            g0 = gens[0]
            # Either a list or a single generation
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
    
    # ---------- Single Model Chat ----------
    async def stream_single_chat(
        self,
        registry: Any,
        wire: str,
        messages: List[Dict[str, str]],
        model_params: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        memory_backend: Optional[Any] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream a single model chat response."""
        if not wire:
            raise HTTPException(400, "Missing 'wire' parameter")
        
        thread_id = thread_id or "default"
        model_params = model_params or {}
        
        _log(f"stream_single_chat: wire={wire}, thread_id={thread_id}")
        
        # Build graph using GraphService
        graph, _ = self.graph_service.build_single_model_graph(
            registry=registry,
            wire=wire,
            model_kwargs=model_params,
            memory_backend=memory_backend,
        )
        
        # Open event
        yield self.sse_event({"type": "open", "scope": "single"}, event="open")
        await asyncio.sleep(0)
        
        config = {"configurable": {"thread_id": thread_id}}
        last_beat = asyncio.get_event_loop().time()
        emitted_any = False
        
        # Sanitize messages
        sanitized = self.sanitize_messages(messages)
        
        async for ev in graph.astream_events({"messages": sanitized, "meta": {}}, config=config):
            # Heartbeat ~10s
            now = asyncio.get_event_loop().time()
            if now - last_beat > 10:
                yield self.sse_comment("hb")
                last_beat = now
            
            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}
            
            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = self.graph_service.extract_text_from_chunk(chunk) if chunk is not None else ""
                if piece:
                    emitted_any = True
                    yield self.sse_event({
                        "type": "delta",
                        "scope": "single",
                        "node": meta.get("langgraph_node"),
                        "delta": piece,
                        "done": False,
                    })
            
            elif et == "on_chat_model_end":
                # Fallback if nothing streamed
                if not emitted_any:
                    final_text = self._extract_final_text(data)
                    if final_text:
                        yield self.sse_event({
                            "type": "delta",
                            "scope": "single", 
                            "node": meta.get("langgraph_node"),
                            "delta": final_text,
                            "done": False,
                        })
            
            elif et == "on_chat_model_error":
                err = data.get("error")
                yield self.sse_event({
                    "type": "error",
                    "scope": "single",
                    "node": meta.get("langgraph_node"),
                    "error": str(err) if err else "Unknown error",
                    "done": False,
                }, event="error")
        
        # Done event
        yield self.sse_event({"type": "done", "scope": "single", "done": True}, event="done")
    
    # ---------- Multi Model Chat ----------
    async def stream_multi_chat(
        self,
        registry: Any,
        targets: List[str],
        messages: List[Dict[str, str]],
        per_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        thread_id: Optional[str] = None,
        memory_backend: Optional[Any] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream multi-model chat responses for comparison."""
        if not targets:
            raise HTTPException(400, "Missing 'targets' list")
        
        thread_id = thread_id or "compare"
        per_model_params = per_model_params or {}
        
        _log(f"stream_multi_chat: targets={targets}, thread_id={thread_id}")
        
        # Build graph using GraphService
        graph, _ = self.graph_service.build_multi_model_graph(
            registry=registry,
            wires=targets,
            per_model_params=per_model_params,
            memory_backend=memory_backend,
        )
        node_to_wire = getattr(graph, "_node_to_wire", {}) or {}
        
        # Open event
        yield self.sse_event({"type": "open", "scope": "multi", "models": targets}, event="open")
        await asyncio.sleep(0)
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Sanitize messages
        sanitized = self.sanitize_messages(messages)
        
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
                yield self.sse_comment("hb")
                last_beat = now
            
            et = ev.get("event")
            data = ev.get("data") or {}
            meta = ev.get("metadata") or {}
            node = meta.get("langgraph_node")
            
            # Try to get a proper wire id for the frontend
            wire = (meta.get("wire") or meta.get("model") or "")
            if not wire and node and node in node_to_wire:
                wire = node_to_wire[node]
            
            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = self.graph_service.extract_text_from_chunk(chunk) if chunk is not None else ""
                if piece and wire:
                    emitted_any[wire] = True
                    yield self.sse_event({
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
                        final_text = self._extract_final_text(data)
                        if final_text:
                            yield self.sse_event({
                                "type": "delta",
                                "scope": "multi",
                                "model": wire,
                                "node": node,
                                "text": final_text,
                                "done": False
                            })
                    
                    # Mark this model as done
                    yield self.sse_event({
                        "type": "done",
                        "scope": "multi",
                        "model": wire,
                        "done": True
                    }, event="done")
            
            elif et == "on_chat_model_error":
                if wire:
                    err = data.get("error")
                    yield self.sse_event({
                        "type": "error",
                        "scope": "multi",
                        "model": wire,
                        "node": node,
                        "error": str(err) if err else "Unknown error",
                        "done": False
                    }, event="error")
        
        # Safety net: Ensure all requested models have a done marker
        for w in targets:
            yield self.sse_event({"type": "done", "scope": "multi", "model": w, "done": True}, event="done")
