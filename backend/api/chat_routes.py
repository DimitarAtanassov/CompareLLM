from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator, Dict, List, Optional
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import asyncio, time, json

from services.enhanced_chat_service import EnhancedChatService, get_enhanced_chat_service
from models.enhanced_requests import EnhancedChatRequest, EnhancedOpenAIChatRequest
from models.responses import ChatResponse


from starlette.responses import StreamingResponse

# ---------- Local response models to preserve OpenAI-compatible schema ----------
class OpenAIChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

# ---------- Model capabilities / validation ----------
class ModelCapabilities(BaseModel):
    model_name: str
    provider_name: str
    provider_type: str
    supports_thinking: Optional[bool] = None
    supports_tools: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    supports_system_messages: Optional[bool] = None
    max_context_tokens: Optional[int] = None
    default_rpm: Optional[int] = None
    default_tpm: Optional[int] = None

class ValidationResult(BaseModel):
    valid: bool
    warnings: List[str]
    errors: List[str]

class ParameterExampleResponse(BaseModel):
    anthropic_example: Dict[str, Any]
    openai_example: Dict[str, Any]
    gemini_example: Dict[str, Any]
    ollama_example: Dict[str, Any]

router = APIRouter()

def _resolve_models(req: EnhancedChatRequest, svc: EnhancedChatService) -> List[str]:
    return req.models or list(svc.registry.model_map.keys())

def _ndjson(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

# ---------- OpenAI-compatible chat completions (single model) ----------
# Mounted at: /v2/chat/completions AND legacy /chat/completions
@router.post("/completions", response_model=OpenAIChatResponse, summary="OpenAI-compatible chat completions")
async def openai_compatible_chat(
    request: EnhancedOpenAIChatRequest,
    svc: EnhancedChatService = Depends(get_enhanced_chat_service),
    req: Request = None,
) -> OpenAIChatResponse:
    # Validate model availability
    reg = getattr(req.app.state, "registry", None) if req else None
    if not reg:
        raise HTTPException(status_code=500, detail="Model registry not initialized")
    if request.model not in reg.model_map:
        available = list(reg.model_map.keys())
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found. Available: {available}")

    # Convert to internal enhanced format
    internal: EnhancedChatRequest = request.to_enhanced_request()

    # Validate request for the specific model
    validation = await svc.validate_request_for_model(request.model, internal)
    if not validation.get("valid", True):
        raise HTTPException(status_code=400, detail="; ".join(validation.get("errors") or ["invalid request"]))

    # Execute
    resp: ChatResponse = await svc.chat_completion(internal)

    # Map back to OpenAI shape
    model_answer = resp.answers.get(request.model)
    content = (model_answer.answer if model_answer and model_answer.answer else "") if hasattr(model_answer, "answer") else ""
    choice = OpenAIChoice(
        index=0,
        message={"role": "assistant", "content": content},
        finish_reason="stop",
    )
    # Rough word-count proxy for tokens
    prompt_tokens = sum(len(m.get("content", "").split()) for m in request.messages)
    completion_tokens = len(content.split())
    return OpenAIChatResponse(
        id=f"enhanced-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[choice],
        usage=OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

# ---------- Enhanced multi-model chat (non-streaming) ----------
@router.post("/completions/enhanced", response_model=ChatResponse, summary="Multi-model chat completion (enhanced)")
async def chat_completions_enhanced(
    body: EnhancedChatRequest,
    svc: EnhancedChatService = Depends(get_enhanced_chat_service),
) -> ChatResponse:
    return await svc.chat_completion(body)

# ---------- Enhanced multi-model chat (streamed NDJSON) ----------
def _ndjson_iter(stream: AsyncIterator[Dict[str, Any]]):
    async def generator():
        async for evt in stream:
            yield (JSONResponse(content=evt).body + b"\n")
    return generator()

@router.post("/completions/enhanced/ndjson", summary="Multi-model chat (NDJSON streaming)")
async def chat_completions_enhanced_stream(
    body: EnhancedChatRequest,
    request: Request,
    svc: EnhancedChatService = Depends(get_enhanced_chat_service),
):
    """
    NDJSON contract the UI expects:
      {"type":"meta","models":[...]}
      {"type":"chunk","model":"<name>","answer":"<delta>","latency_ms":123}
      ... (many lines)
      {"type":"done"}
    """

    # Resolve and validate models up-front
    models = _resolve_models(body, svc)
    if not models:
        raise HTTPException(status_code=400, detail="No models specified and none available")

    # Small helper to stream in NDJSON with meta/keepalive/done
    async def gen() -> AsyncIterator[bytes]:
        # 1) meta (IMMEDIATE)
        yield _ndjson({"type": "meta", "models": models})

        # 2) stream deltas from service; also send keep-alives if quiet
        last_sent = time.monotonic()
        KEEPALIVE_SEC = 10.0

        # We use the existing service fan-in stream. It already:
        #   - runs all models in parallel
        #   - yields {"model": str, "delta": str, "latency_ms": int}
        #   - yields "[error] ..." in delta for provider errors
        stream = svc.stream_answers(body)

        async def keepalive():
            # Emit minimal heartbeats while stream is active but quiet
            nonlocal last_sent
            try:
                while True:
                    await asyncio.sleep(KEEPALIVE_SEC)
                    if time.monotonic() - last_sent >= KEEPALIVE_SEC:
                        yield _ndjson({"type": "chunk", "model": None, "answer": "", "latency_ms": 0})
                        last_sent = time.monotonic()
                    else:
                        # no-op; recent data already flowed
                        pass
            except asyncio.CancelledError:
                return

        # Run keepalive and stream concurrently (and stop KA once stream ends)
        ka_queue: asyncio.Queue[bytes] = asyncio.Queue()
        ka_task: Optional[asyncio.Task] = None

        async def ka_runner():
            async for tick in keepalive():
                await ka_queue.put(tick)

        ka_task = asyncio.create_task(ka_runner())

        async def ka_drain():
            while not ka_queue.empty():
                yield await ka_queue.get()

        try:
            async for evt in stream:
                # Drain any pending keepalives first (avoid long gaps)
                async for tick in ka_drain():
                    yield tick

                # Map the event to UI schema
                # evt = {"model": "...", "delta": "...", "latency_ms": int}
                model = evt.get("model")
                delta = evt.get("delta", "")
                latency_ms = int(evt.get("latency_ms", 0))

                if isinstance(delta, str) and delta.startswith("[error]"):
                    yield _ndjson({"type": "chunk", "model": model, "error": delta, "latency_ms": latency_ms})
                else:
                    yield _ndjson({"type": "chunk", "model": model, "answer": delta or "", "latency_ms": latency_ms})

                last_sent = time.monotonic()

        finally:
            if ka_task:
                ka_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ka_task

        # 3) done (ALWAYS)
        yield _ndjson({"type": "done"})

    # NOTE: Important headers to avoid proxy buffering and caching
    headers = {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",  # disable nginx buffering if present
    }
    return StreamingResponse(gen(), headers=headers)

# ---------- Back-compat alias for historical client ----------
@router.post("/ask/ndjson", include_in_schema=False)
async def legacy_ask_ndjson(
    body: EnhancedChatRequest,
    svc: EnhancedChatService = Depends(get_enhanced_chat_service),
):
    stream = svc.stream_answers(body)
    return StreamingResponse(_ndjson_iter(stream), media_type="application/x-ndjson")

# ---------- Anthropic-optimized endpoint (mounted at /v2/chat/anthropic and /chat/anthropic) ----------
@router.post("/anthropic", response_model=ChatResponse)
async def anthropic_optimized_chat(
    messages: List[Dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 8192,
    temperature: float = 1.0,
    thinking_enabled: bool = False,
    thinking_budget: Optional[int] = None,
    service_tier: str = "auto",
    stop_sequences: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    svc: EnhancedChatService = Depends(get_enhanced_chat_service),
) -> ChatResponse:
    # Build an EnhancedChatRequest inline (no UI required)
    req = EnhancedChatRequest(
        models=[model],
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        anthropic_params={{  # matches your Anthropic params model
            "thinking_enabled": thinking_enabled,
            "thinking_budget_tokens": thinking_budget,
            "service_tier": service_tier,
            "stop_sequences": stop_sequences,
            "top_k": top_k,
            "top_p": top_p,
        }},
    )
    return await svc.chat_completion(req)

# ---------- Capabilities / validation / parameter examples ----------
@router.get("/models/{model_name}/capabilities", response_model=ModelCapabilities)
async def get_model_capabilities(model_name: str, req: Request, svc: EnhancedChatService = Depends(get_enhanced_chat_service)):
    try:
        caps = svc.get_model_capabilities(model_name)
        return ModelCapabilities(**caps)
    except Exception as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=500, detail=msg)

@router.post("/models/{model_name}/validate", response_model=ValidationResult)
async def validate_request_for_model(model_name: str, body: EnhancedChatRequest, svc: EnhancedChatService = Depends(get_enhanced_chat_service)):
    out = await svc.validate_request_for_model(model_name, body)
    return ValidationResult(**out)

@router.get("/parameters/examples", response_model=ParameterExampleResponse)
async def get_parameter_examples():
    return ParameterExampleResponse(
        anthropic_example={
            "thinking_enabled": True,
            "thinking_budget_tokens": 2048,
            "top_k": 40,
            "top_p": 0.9,
            "stop_sequences": ["Human:", "Assistant:"],
            "service_tier": "auto",
            "tool_choice_type": "auto",
            "user_id": "user-123",
        },
        openai_example={
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["Human:", "AI:"],
            "seed": 42,
            "response_format": {"type": "json_object"},
            "user": "user-123",
        },
        gemini_example={
            "top_k": 40,
            "top_p": 0.9,
            "candidate_count": 1,
            "stop_sequences": ["Human:", "AI:"],
            "safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
        },
        ollama_example={
            "mirostat": 1,
            "mirostat_eta": 0.1,
            "mirostat_tau": 5.0,
            "num_ctx": 4096,
            "repeat_last_n": 64,
            "repeat_penalty": 1.1,
            "seed": 42,
            "top_k": 40,
            "top_p": 0.9,
            "format": "json",
        },
    )
