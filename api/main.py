from __future__ import annotations
import json
import os, asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import httpx
import yaml

@dataclass
class Provider:
    name: str
    type: str                 
    base_url: str
    api_key_env: Optional[str]
    headers: Dict[str, str]
    models: List[str]

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env) if self.api_key_env else None

class ModelRegistry:
    def __init__(self, cfg: Dict[str, Any]):
        self.providers: Dict[str, Provider] = {}
        for pname, p in cfg.get("providers", {}).items():
            self.providers[pname] = Provider(
                name=pname,
                type=p["type"],
                base_url=p["base_url"].rstrip("/"),
                api_key_env=(p.get("api_key_env") or None),
                headers=(p.get("headers") or {}),
                models=(p.get("models") or []),
            )
        # map model -> (provider, model_name)
        self.model_map: Dict[str, Tuple[Provider, str]] = {}
        for prov in self.providers.values():
            for m in prov.models:
                self.model_map[m] = (prov, m)

    @classmethod
    def from_path(cls, path: str) -> "ModelRegistry":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(cfg)

# ------------ per-host serialization (helps on Mac CPU) ------------
_HOST_LOCKS: Dict[str, asyncio.Semaphore] = {}
def host_lock_key(provider: Provider) -> str:
    return provider.base_url
def host_lock(provider: Provider) -> asyncio.Semaphore:
    k = host_lock_key(provider)
    if k not in _HOST_LOCKS:
        _HOST_LOCKS[k] = asyncio.Semaphore(1)
    return _HOST_LOCKS[k]

# -------------------------- Chat adapters --------------------------

async def chat_call(
    provider: Provider,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    min_tokens: Optional[int] = None,    # <— added
    timeout_s: int = 180,
) -> str:
    """
    Unified chat call for: OpenAI-compatible providers + Gemini.
    """
    headers = dict(provider.headers or {})

    if provider.type == "openai":
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        url = f"{provider.base_url}/chat/completions"

        def build_payload(use_completion_tokens: bool, include_temperature: bool):
            body = {
                "model": model,
                "messages": messages,
            }
            if include_temperature:
                body["temperature"] = temperature
            if use_completion_tokens:
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens
            # no min_tokens equivalent here
            return body
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            use_completion = False           # start with max_tokens
            include_temp   = True            # start including temperature

            # attempt 1
            r = await client.post(
                url, headers=headers, json=build_payload(use_completion, include_temp)
            )

            # If bad request, inspect and adapt up to two times
            for _ in range(2):
                if r.status_code != 400:
                    break
                try:
                    err = r.json()
                    # OpenAI errors may be {"error": {...}} OR flat dict
                    meta = err.get("error", err)
                    msg = str(meta.get("message", meta))
                except Exception:
                    msg = ""

                need_completion = ("max_tokens" in msg and "max_completion_tokens" in msg)
                temp_unsupported = ("temperature" in msg and "supported" in msg)

                if not (need_completion or temp_unsupported):
                    break  # unknown 400—let _raise_nice handle it

                # adjust knobs based on the message
                if need_completion:
                    use_completion = True
                if temp_unsupported:
                    include_temp = False

                # retry with adjusted payload
                r = await client.post(
                    url, headers=headers, json=build_payload(use_completion, include_temp)
                )

            _raise_nice(r)
            data = r.json()
            return data["choices"][0]["message"]["content"]

    elif provider.type == "gemini":
        # Gemini: models/*:generateContent
        # Convert OpenAI-style messages -> Gemini contents.
        # Prefer using systemInstruction if present.
        sys_texts = [m["content"] for m in messages if m["role"] == "system"]
        user_assistant_msgs = [m for m in messages if m["role"] in ("user", "assistant")]

        contents = []
        for m in user_assistant_msgs:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        params = {"key": provider.api_key} if provider.api_key else {}
        url = f"{provider.base_url}/v1beta/models/{model}:generateContent"
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if sys_texts:
            payload["systemInstruction"] = {"role": "user", "parts": [{"text": "\n".join(sys_texts)}]}

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, params=params, headers=headers, json=payload)
            _raise_nice(r)
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    elif provider.type == "anthropic":
        # Anthropic Messages API: POST /v1/messages
        # Required headers:
        #   - x-api-key: <ANTHROPIC_API_KEY>
        #   - anthropic-version: e.g. "2023-06-01"
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        # Allow overriding via config headers; otherwise set a sane default.
        headers.setdefault("anthropic-version", "2023-06-01")

        url = f"{provider.base_url.rstrip('/')}/v1/messages"

        # Extract system message (concat if multiple; order preserved)
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        system_str = "\n".join(system_texts) if system_texts else None

        # Translate the remaining turns
        turns = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            # Anthropic expects content as array of parts; simplest is one text part
            turns.append({
                "role": "user" if role == "user" else "assistant",
                "content": [{"type": "text", "text": m.get("content", "")}],
            })

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system_str:
            payload["system"] = system_str
        if min_tokens is not None:
            payload["min_output_tokens"] = int(min_tokens)

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            _raise_nice(r)
            data = r.json()

            # data.content is a list of blocks; collect any text blocks
            parts = data.get("content", []) or []
            out: List[str] = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text" and "text" in p:
                    out.append(p["text"])
            return "".join(out) if out else json.dumps(data)

    else:
        raise ValueError(f"Unsupported provider type: {provider.type}")


def _raise_nice(r: httpx.Response) -> None:
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            j = r.json()
            detail = j.get("error", j.get("message", "")) or str(j)
        except Exception:
            pass
        raise RuntimeError(f"{r.status_code} {r.reason_phrase}: {detail}") from e

import os, asyncio
from fastapi import FastAPI, HTTPException
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict  # ✅ use pydantic.Field
from registry import ModelRegistry, chat_call, host_lock
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import time

# ------------------------ logging (added) ------------------------
import logging

LOG = logging.getLogger("askmanyllms")
if not LOG.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))

def log_event(kind: str, **fields):
    """
    Single-line JSON logs for easy parsing in Docker/Cloud logs.
    """
    try:
        LOG.info(json.dumps({"event": kind, **fields}, ensure_ascii=False))
    except Exception as _:
        # Fallback to a simple message if something isn’t JSON-serializable
        LOG.info(f"{kind} | {fields}")
# ---------------------------------------------------------------

CFG_PATH = os.getenv("MODELS_CONFIG", "/config/models.yaml")
REGISTRY = ModelRegistry.from_path(CFG_PATH)

app = FastAPI(title="Ask Many Models (Pluggable)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str
    models: List[str] | None = None
    # global defaults
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8192
    min_tokens: Optional[int] = None
    # per-model overrides
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")

@app.get("/providers")
def providers():
    return {
        "providers": [
            {
                "name": p.name,
                "type": p.type,
                "base_url": p.base_url,
                "models": p.models,
                "auth_required": bool(p.api_key_env),
            } for p in REGISTRY.providers.values()
        ]
    }

@app.post("/ask")
async def ask(req: ChatRequest):
    messages = [
        {"role": "system", "content": "Answer clearly and concisely."},
        {"role": "user", "content": req.prompt},
    ]
    chosen = req.models or list(REGISTRY.model_map.keys())
    unknown = [m for m in chosen if m not in REGISTRY.model_map]
    if unknown:
        raise HTTPException(400, f"Unknown models: {unknown}")

    async def one(model_name: str):
        provider, model = REGISTRY.model_map[model_name]

        # ----- logging: record params passed to this model -----
        log_event(
            "call.start",
            route="/ask",
            provider=provider.name,
            provider_type=provider.type,
            model=model_name,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            min_tokens=req.min_tokens,
        )
        start = time.perf_counter()
        # -------------------------------------------------------

        # serialize per-host (helps on Mac/CPU + Ollama)
        try:
            async with host_lock(provider):
                result = await chat_call(provider, model, messages, req.temperature, req.max_tokens)
            dur = int((time.perf_counter() - start) * 1000)
            log_event(
                "call.end",
                route="/ask",
                provider=provider.name,
                provider_type=provider.type,
                model=model_name,
                ok=True,
                duration_ms=dur,
                answer_chars=len(result or ""),
            )
            return result
        except Exception as e:
            dur = int((time.perf_counter() - start) * 1000)
            log_event(
                "call.end",
                route="/ask",
                provider=provider.name,
                provider_type=provider.type,
                model=model_name,
                ok=False,
                duration_ms=dur,
                error=str(e),
            )
            raise

    tasks = [asyncio.create_task(one(m)) for m in chosen]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    answers: Dict[str, Any] = {}
    for m, res in zip(chosen, results):
        answers[m] = {"error": str(res)} if isinstance(res, Exception) else {"answer": res}
    return {"prompt": req.prompt, "models": chosen, "answers": answers}

# /ask/ndjson
@app.post("/ask/ndjson")
async def ask_ndjson(req: ChatRequest):
    messages = [
        {"role": "system", "content": "Answer clearly and concisely."},
        {"role": "user", "content": req.prompt},
    ]
    chosen = req.models or list(REGISTRY.model_map.keys())
    unknown = [m for m in chosen if m not in REGISTRY.model_map]
    if unknown:
        raise HTTPException(400, f"Unknown models: {unknown}")

    def merged_params(m: str):
        mp = req.model_params.get(m, {})
        temperature = float(mp.get("temperature", req.temperature if req.temperature is not None else 0.7))
        max_tokens = int(mp.get("max_tokens", req.max_tokens if req.max_tokens is not None else 8192))
        min_tokens = mp.get("min_tokens", req.min_tokens)
        min_tokens = int(min_tokens) if (min_tokens is not None) else None
        return temperature, max_tokens, min_tokens

    async def gen(): 
        yield json.dumps({"type": "meta", "models": chosen}) + "\n"
        out_q: asyncio.Queue[str] = asyncio.Queue()

        async def run_one(model_name: str):
            provider, model = REGISTRY.model_map[model_name]
            start = time.perf_counter()
            try:
                t, mx, mn = merged_params(model_name)

                # ----- logging: record per-model params -----
                log_event(
                    "call.start",
                    route="/ask/ndjson",
                    provider=provider.name,
                    provider_type=provider.type,
                    model=model_name,
                    temperature=t,
                    max_tokens=mx,
                    min_tokens=mn,
                )
                # -------------------------------------------

                async with host_lock(provider):
                    final_text = await chat_call(provider, model, messages, t, mx, mn)
                ms = int((time.perf_counter() - start) * 1000)

                log_event(
                    "call.end",
                    route="/ask/ndjson",
                    provider=provider.name,
                    provider_type=provider.type,
                    model=model_name,
                    ok=True,
                    duration_ms=ms,
                    answer_chars=len(final_text or ""),
                )

                out_q.put_nowait(json.dumps({
                    "type": "chunk",
                    "model": model_name,
                    "answer": final_text,
                    "latency_ms": ms
                }) + "\n")
            except Exception as e:
                ms = int((time.perf_counter() - start) * 1000)

                log_event(
                    "call.end",
                    route="/ask/ndjson",
                    provider=provider.name,
                    provider_type=provider.type,
                    model=model_name,
                    ok=False,
                    duration_ms=ms,
                    error=str(e),
                )

                out_q.put_nowait(json.dumps({
                    "type": "chunk",
                    "model": model_name,
                    "error": str(e),
                    "latency_ms": ms
                }) + "\n")

        tasks = [asyncio.create_task(run_one(m)) for m in chosen]
        pending = set(tasks)
        while pending or not out_q.empty():
            try:
                item = await asyncio.wait_for(out_q.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
                pending = {t for t in tasks if not t.done()}
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
