from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------- Schemas ----------

class ModelSelection(BaseModel):
    provider: str = Field(..., description="Provider key from models.yaml (e.g., 'openai')")
    model: str    = Field(..., description="Model name exactly as in models.yaml (e.g., 'gpt-4o')")

class ChatMessageIn(BaseModel):
    role: str
    content: str

class ProviderAnthropicParams(BaseModel):
    thinking_enabled: Optional[bool] = None
    thinking_budget_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None

class ProviderOpenAIParams(BaseModel):
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None

class ProviderGeminiParams(BaseModel):
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    candidate_count: Optional[int] = None
    safety_settings: Optional[List[Any]] = None

class ProviderOllamaParams(BaseModel):
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    repeat_penalty: Optional[float] = None

class ProviderCohereParams(BaseModel):
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = Field(default=None, ge=0, le=18446744073709552000)
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: Optional[float]  = Field(default=None, ge=0.0, le=1.0)
    k: Optional[int] = Field(default=None, ge=0, le=500)
    p: Optional[float] = Field(default=None, ge=0.01, le=0.99)
    logprobs: Optional[bool] = None
    #raw_prompting: Optional[bool] = None

class ProviderCerebrasParams(BaseModel):
    # OpenAI-wire compatible knobs
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None

    # JSON mode (OpenAI-style): {"type": "json_object"}
    response_format: Optional[Dict[str, Any]] = None

    # Tool use (if you bind tools in your adapter): "auto" | "none" | "any" | <tool name>
    tool_choice: Optional[str] = None

class ProviderDeepseekParams(BaseModel):
    frequency_penalty:Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    logprobs: Optional[bool]  = None
    top_logprobs: Optional[int]   = Field(default=None, ge=0, le=20)

class ChatRequest(BaseModel):
    prompt: str
    selections: List[ModelSelection]
    history: Optional[List[ChatMessageIn]] = None
    system: Optional[str] = None

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None

    anthropic_params: Optional[ProviderAnthropicParams] = None
    openai_params: Optional[ProviderOpenAIParams] = None
    gemini_params: Optional[ProviderGeminiParams] = None
    ollama_params: Optional[ProviderOllamaParams] = None
    cohere_params: Optional[ProviderCohereParams] = None
    deepseek_params: Optional[ProviderDeepseekParams] = None
    cerebras_params: Optional[ProviderCerebrasParams] = None

    model_params: Optional[Dict[str, Dict[str, Any]]] = None

# ---------- Helpers ----------

def _history_to_messages(history: Optional[List[ChatMessageIn]]) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    if not history:
        return msgs
    for m in history:
        role = (m.role or "").lower().strip()
        if role in ("user", "human"):
            msgs.append(HumanMessage(content=m.content))
        elif role in ("assistant", "ai"):
            msgs.append(AIMessage(content=m.content))
        elif role == "system":
            msgs.append(SystemMessage(content=m.content))
        else:
            msgs.append(HumanMessage(content=m.content))
    return msgs

def _build_prompt(system_text: Optional[str]) -> ChatPromptTemplate:
    msgs: List[Any] = []
    if system_text:
        msgs.append(("system", system_text))
    msgs.append(MessagesPlaceholder(variable_name="history"))
    msgs.append(("human", "{input}"))
    return ChatPromptTemplate.from_messages(msgs)

def _inputs(req: ChatRequest) -> Dict[str, Any]:
    inputs = {"history": _history_to_messages(req.history), "input": req.prompt}
    print(f"[inputs] built inputs -> {inputs}")
    return inputs

def _reg_get(request: Request, provider: str, model: str):
    reg = request.app.state.registry
    key = f"{provider}:{model}"
    if key not in reg:
        raise HTTPException(status_code=404, detail=f"Model not found: {key}")
    return reg.get(provider, model)

def _merge_params(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if extra:
        for k, v in extra.items():
            if v is not None:
                out[k] = v
    return out

# chat.py

def _bind_model(sel: ModelSelection, chat_model: Any, req: ChatRequest) -> Any:
    common: Dict[str, Any] = {}
    if req.temperature is not None:
        common["temperature"] = req.temperature
    if req.max_tokens is not None:
        common["max_tokens"] = req.max_tokens
    if req.min_tokens is not None:
        common["min_tokens"] = req.min_tokens

    provider_group: Dict[str, Any] = {}

    if sel.provider == "openai" and req.openai_params:
        provider_group.update(req.openai_params.model_dump(exclude_none=True))

    elif sel.provider == "anthropic" and req.anthropic_params:
        ap = req.anthropic_params
        if ap.thinking_enabled:
            thinking = {"type": "enabled"}
            if ap.thinking_budget_tokens:
                thinking["budget_tokens"] = ap.thinking_budget_tokens
            provider_group["thinking"] = thinking
        for k in ("top_k", "top_p", "stop_sequences"):
            v = getattr(ap, k)
            if v is not None:
                provider_group[k] = v

    elif sel.provider == "gemini" and req.gemini_params:
        gp = req.gemini_params.model_dump(exclude_none=True)
        if "max_tokens" in common:
            provider_group["max_output_tokens"] = common.pop("max_tokens")
        provider_group.update(gp)

    elif sel.provider == "ollama" and req.ollama_params:
        ok = req.ollama_params.model_dump(exclude_none=True)
        if ok:
            provider_group["model_kwargs"] = ok

    elif sel.provider == "cohere" and req.cohere_params:
        provider_group.update(req.cohere_params.model_dump(exclude_none=True))
    
    elif sel.provider == "deepseek" and req.deepseek_params:
        provider_group.update(req.deepseek_params.model_dump(exclude_none=True))

    elif sel.provider == "cerebras" and req.cerebras_params:
        cp = req.cerebras_params.model_dump(exclude_none=True)
        # If the frontend ever sends "stop" as List[str], pass through;
        # both ChatCerebras and OpenAI-wire accept it.
        provider_group.update(cp)

    # Per-model overrides (your existing pattern)
    per_model = (req.model_params or {}).get(sel.model) or {}

    # Gemini rename already handled above; nothing special for Cohere

    bound_kwargs = _merge_params(_merge_params(common, provider_group), per_model)

    print(f"[bind_model] provider={sel.provider}, model={sel.model}")
    print(f"    common={common}")
    print(f"    provider_group={provider_group}")
    print(f"    per_model={per_model}")
    print(f"    FINAL bound_kwargs={bound_kwargs}")

    return chat_model.bind(**bound_kwargs)


def _make_chain(chat_model: Any, prompt_tmpl: ChatPromptTemplate):
    return prompt_tmpl | chat_model | StrOutputParser()


# ---------- Batch ----------

@router.post("/batch")
async def chat_batch(req: ChatRequest, request: Request):
    prompt_tmpl = _build_prompt(req.system)
    inputs = _inputs(req)

    async def _one(sel: ModelSelection):
        base_model = _reg_get(request, sel.provider, sel.model)
        model = _bind_model(sel, base_model, req)
        chain = _make_chain(model, prompt_tmpl)
        print(f"[chat_batch] invoking {sel.provider}:{sel.model} with inputs={inputs}")
        try:
            if hasattr(chain, "ainvoke"):
                text = await chain.ainvoke(inputs)
            else:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(None, lambda: chain.invoke(inputs))
            print(f"[chat_batch] ✅ {sel.provider}:{sel.model} -> {text[:100]}...")
            return {"provider": sel.provider, "model": sel.model, "response": text}
        except Exception as e:
            print(f"[chat_batch] ❌ {sel.provider}:{sel.model} -> {e}")
            raise

    results = await asyncio.gather(*[_one(s) for s in req.selections], return_exceptions=True)

    out = []
    for sel, r in zip(req.selections, results):
        if isinstance(r, Exception):
            out.append({"provider": sel.provider, "model": sel.model, "error": f"{type(r).__name__}: {r}"})
        else:
            out.append(r)
    return {"results": out}


# ---------- Stream ----------

@router.post("/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """
    Minimal, provider-agnostic streamer using LangChain's astream().
    - Sends {start, delta, end} per model + final all_done.
    - No special-casing for providers (Gemini included).
    - Falls back to ainvoke()/invoke() if astream is not available.
    """
    prompt_tmpl = _build_prompt(req.system)
    inputs = _inputs(req)
    queue: asyncio.Queue[str] = asyncio.Queue()

    def _normalize_provider(p: str) -> str:
        p = (p or "").lower().strip()
        # Accept "google" et al. but normalize to "gemini"
        if p in ("google", "googleai", "google_genai", "gemini"):
            return "gemini"
        return p

    async def producer(sel: ModelSelection):
        prov = _normalize_provider(sel.provider)
        model_name = sel.model

        # START event first so UI can render a slot immediately
        await queue.put(json.dumps({"type": "start", "provider": prov, "model": model_name}) + "\n")

        # Resolve and bind
        try:
            base_model = _reg_get(request, prov, model_name)
        except HTTPException as e:
            await queue.put(json.dumps({
                "type": "error", "provider": prov, "model": model_name,
                "error": f"Model not found: {prov}:{model_name}"
            }) + "\n")
            await queue.put(json.dumps({"type": "end", "provider": prov, "model": model_name}) + "\n")
            return

        try:
            model = _bind_model(ModelSelection(provider=prov, model=model_name), base_model, req)
            chain = _make_chain(model, prompt_tmpl)  # prompt | model | StrOutputParser()
        except Exception as e:
            await queue.put(json.dumps({
                "type": "error", "provider": prov, "model": model_name,
                "error": f"bind/chain error: {type(e).__name__}: {e}"
            }) + "\n")
            await queue.put(json.dumps({"type": "end", "provider": prov, "model": model_name}) + "\n")
            return

        # Stream if possible; otherwise fall back to one-shot
        try:
            streamed_any = False

            if hasattr(chain, "astream"):
                async for chunk in chain.astream(inputs):
                    # StrOutputParser returns strings; chat models may return AIMessageChunk
                    text = getattr(chunk, "content", chunk)
                    if not isinstance(text, str):
                        text = str(text)
                    if not text:
                        continue
                    streamed_any = True
                    await queue.put(json.dumps({
                        "type": "delta", "provider": prov, "model": model_name, "text": text
                    }) + "\n")

            if not streamed_any:
                # Fallback: produce one final delta
                if hasattr(chain, "ainvoke"):
                    text = await chain.ainvoke(inputs)
                else:
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(None, lambda: chain.invoke(inputs))
                text = getattr(text, "content", text)
                if not isinstance(text, str):
                    text = str(text)
                await queue.put(json.dumps({
                    "type": "delta", "provider": prov, "model": model_name, "text": text or ""
                }) + "\n")

        except Exception as e:
            await queue.put(json.dumps({
                "type": "error", "provider": prov, "model": model_name,
                "error": f"{type(e).__name__}: {e}"
            }) + "\n")

        finally:
            # Always emit a terminal 'end' for each model
            await queue.put(json.dumps({"type": "end", "provider": prov, "model": model_name}) + "\n")

    async def supervisor():
        tasks = [asyncio.create_task(producer(s)) for s in req.selections]
        await asyncio.gather(*tasks, return_exceptions=True)
        await queue.put(json.dumps({"type": "all_done"}) + "\n")

    async def streamer():
        super_task = asyncio.create_task(supervisor())
        try:
            while True:
                line = await queue.get()
                yield line
                if line.strip() in ('{"type":"all_done"}', '{"type": "all_done"}'):
                    break
        finally:
            super_task.cancel()

    return StreamingResponse(streamer(), media_type="application/x-ndjson")


