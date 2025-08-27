# backend/adapters/enhanced_chat_adapter.py
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
import json

from core.exceptions import ProviderError
from providers.base import Provider
from providers.anthropic_params import AnthropicParameters, create_anthropic_params, get_model_config

# Per-host locks for serialization
_HOST_LOCKS: Dict[str, asyncio.Semaphore] = {}

def get_host_lock(provider: Provider) -> asyncio.Semaphore:
    """Get or create a semaphore for the provider's host."""
    if provider.base_url not in _HOST_LOCKS:
        _HOST_LOCKS[provider.base_url] = asyncio.Semaphore(1)
    return _HOST_LOCKS[provider.base_url]


def _get_wire(provider: Provider) -> str:
    """
    Return the wire protocol for a provider, defaulting to provider.type.
    Example values: 'openai', 'anthropic', 'gemini', 'ollama'.
    """
    wire = getattr(provider, "wire", None)
    if wire:
        return str(wire).lower()
    ptype = getattr(provider, "type", None)
    return str(ptype or "").lower()


class EnhancedChatAdapter:
    """Enhanced adapter for chat completion with provider-specific parameter support."""

    async def chat_completion(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,  # Anthropic-only
        timeout_s: int = 180,
        provider_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate chat completion using the appropriate provider with enhanced parameters.

        NOTE: temperature/max_tokens/min_tokens are OPTIONAL. If None, they are NOT sent to the provider,
        allowing the provider's own defaults to apply.
        """
        try:
            print(f"🚀 Enhanced Chat request - Provider: {provider.name}, Model: {model}")
            print(f"🔑 API Key present: {bool(getattr(provider, 'api_key', None))}")
            print(f"🌐 Base URL: {provider.base_url}")
            print(f"📝 Messages count: {len(messages)}")
            if provider_params:
                print(f"⚙️  Provider params: {json.dumps(provider_params, indent=2)}")

            wire = _get_wire(provider)

            if wire == "openai":
                # Any OpenAI-compatible provider (OpenAI, DeepSeek, Together, Groq, etc.)
                return await self._openai_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s, provider_params
                )
            elif wire == "gemini":
                return await self._gemini_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s, provider_params
                )
            elif wire == "anthropic":
                return await self._anthropic_chat_enhanced(
                    provider, model, messages, temperature, max_tokens, min_tokens, timeout_s, provider_params
                )
            elif wire == "ollama":
                return await self._ollama_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s, provider_params
                )
            else:
                raise ProviderError(provider.name, f"Unsupported provider wire: {wire or getattr(provider,'type',None)}")

        except Exception as e:
            print(f"❌ Chat error - Provider: {provider.name}, Model: {model}, Error: {str(e)}")
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(provider.name, str(e))

    # ---------------- Anthropic ----------------

    async def _anthropic_chat_enhanced(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        min_tokens: Optional[int],
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle Anthropic chat completion with full parameter support (omitting unset keys)."""

        # Build kwargs for create_anthropic_params ONLY with set values
        ap_kwargs: Dict[str, Any] = dict(provider_params or {})
        if max_tokens is not None:
            ap_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            ap_kwargs["temperature"] = temperature

        anthropic_params: AnthropicParameters = create_anthropic_params(
            model=model,
            **ap_kwargs
        )

        print(f"🤖 Using Anthropic parameters: {anthropic_params.model_dump(exclude_unset=True)}")

        # Model config diagnostics
        model_config = get_model_config(model)
        if model_config:
            if anthropic_params.thinking and not model_config.supports_thinking:
                print(f"⚠️  Warning: Model {model} may not support extended thinking")
            print(f"📊 Model info: Context={model_config.max_context_tokens}, RPM={model_config.default_rpm}")

        # Headers
        headers = dict(provider.headers or {})
        if getattr(provider, "api_key", None):
            headers["x-api-key"] = provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")

        url = f"{provider.base_url.rstrip('/')}/v1/messages"

        # Extract system
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        system_str = "\n".join(system_texts) if system_texts else None

        # Convert messages
        anthropic_messages = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            anthropic_messages.append({
                "role": "user" if role == "user" else "assistant",
                "content": [{"type": "text", "text": m.get("content", "")}],
            })

        # Anthropic-only extra params
        extra_params: Dict[str, Any] = {}
        if min_tokens is not None:
            extra_params["min_output_tokens"] = int(min_tokens)

        try:
            payload = anthropic_params.to_anthropic_payload(
                messages=anthropic_messages,
                system=system_str
            )
            payload.update(extra_params)

            # If thinking is enabled and user PROVIDED a temperature that's not 1.0, normalize to 1.0.
            thinking = payload.get("thinking")
            ttype = thinking.get("type") if isinstance(thinking, dict) else None
            if ttype == "enabled" and "temperature" in payload and payload.get("temperature") != 1.0:
                payload["temperature"] = 1.0

            print(f"📤 Anthropic payload: {json.dumps(payload, indent=2)}")

        except Exception as e:
            print(f"❌ Error creating Anthropic payload: {e}")
            raise ProviderError(provider.name, f"Invalid parameters: {e}")

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            try:
                r = await client.post(url, headers=headers, json=payload)
                self._ensure_ok_stream(r)
                data = r.json()

                print(f"✅ Anthropic response keys: {list(data.keys())}")

                # Extract text from content blocks
                parts = data.get("content", []) or []
                out: List[str] = []
                for p in parts:
                    if isinstance(p, dict) and p.get("type") == "text" and "text" in p:
                        out.append(p["text"])

                return "".join(out) if out else ""

            except httpx.HTTPStatusError as e:
                error_detail = ""
                try:
                    error_json = r.json()
                    error_detail = error_json.get("error", {}).get("message", r.text)
                except Exception:
                    error_detail = r.text
                print(f"❌ Anthropic HTTP error: {e.response.status_code} - {error_detail}")
                raise ProviderError(provider.name, f"HTTP {e.response.status_code}: {error_detail}")

    # ---------------- OpenAI (and OpenAI-compatible) ----------------
    @staticmethod
    def _sanitize_openai_payload(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove/normalize fields based on model family (gpt-4, gpt-4o, gpt-5, o1-preview, etc.)
        """
        ml = model.lower()
        is_gpt5 = ml.startswith("gpt-5") or ml.startswith("o1-") or ml.startswith("o2-")
        is_gpt4o = ml.startswith("gpt-4o") or ml.startswith("gpt-4.1")

        # gpt-5 / o-series: uses max_completion_tokens, ignores temperature ≠ 1.0
        if is_gpt5:
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
            if payload.get("temperature") not in (None, 1.0):
                payload["temperature"] = 1.0
            payload.pop("response_format", None)
            payload.pop("seed", None)
            if not payload.get("tools"):
                payload.pop("tools", None)
            if "tool_choice" in payload and "tools" not in payload:
                payload.pop("tool_choice")

        # gpt-4o: allow normal temperature/max_tokens
        if is_gpt4o:
            if "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")

        return payload

    # --- OpenAI non-streaming ---
    async def _openai_chat(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
        headers["Content-Type"] = "application/json"

        url = f"{provider.base_url.rstrip('/')}/chat/completions"

        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)  # sanitizer will flip for gpt-5

        # ✅ merge safe OpenAI params (supports either {"openai_params": {...}} or flat)
        safe = ("top_p","frequency_penalty","presence_penalty","stop","tools","tool_choice","user")
        src = (provider_params or {}).get("openai_params", provider_params or {})
        for k in safe:
            if k in src and src[k] is not None:
                payload[k] = src[k]

        # ✅ coerce types to avoid 400s
        if "top_p" in payload:               payload["top_p"] = float(payload["top_p"])
        if "frequency_penalty" in payload:   payload["frequency_penalty"] = float(payload["frequency_penalty"])
        if "presence_penalty" in payload:    payload["presence_penalty"] = float(payload["presence_penalty"])
        if "stop" in payload and isinstance(payload["stop"], str):
            payload["stop"] = [payload["stop"]]

        # ✅ sanitize by family
        payload = self._sanitize_openai_payload(model, payload)
        unsupported_for_cerebras = {"frequency_penalty", "presence_penalty", "logit_bias", "parallel_tool_calls", "service_tier"}

        is_cerebras = getattr(provider, "type", "").lower() == "cerebras" or "cerebras" in provider.base_url
        if is_cerebras and isinstance(payload, dict):
            payload = {k: v for k, v in payload.items() if k not in unsupported_for_cerebras}
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            # 🔁 non-stream: use raise_for_status, not ensure_ok_stream
            self._raise_for_status(r)
            data = r.json()
            return data["choices"][0]["message"]["content"]


    # --- OpenAI streaming ---
    async def _openai_chat_stream(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
        headers["Accept"] = "text/event-stream"
        headers["Content-Type"] = "application/json"

        url = f"{provider.base_url.rstrip('/')}/chat/completions"

        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            if model.startswith("gpt-5"):
                payload["temperature"] = 1.0  # normalized
            else:
                payload["temperature"] = temperature
        if max_tokens is not None:
            if model.startswith("gpt-5"):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        if "top_p" in payload:               payload["top_p"] = float(payload["top_p"])
        if "frequency_penalty" in payload:   payload["frequency_penalty"] = float(payload["frequency_penalty"])
        if "presence_penalty" in payload:    payload["presence_penalty"] = float(payload["presence_penalty"])
        if "stop" in payload and isinstance(payload["stop"], str):
            payload["stop"] = [payload["stop"]]

        payload = self._sanitize_openai_payload(model, payload)
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            self._ensure_ok_stream(r)
            data = r.json()
            return data["choices"][0]["message"]["content"]



    # ---------------- Gemini ----------------

    async def _gemini_chat(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Google Generative Language API (AI Studio) chat via :generateContent.
        Builds spec-compliant payload and surfaces clear errors.
        """
        # ---- 1) Normalize model names (legacy → current), allow -latest alias ----
        remap = {
            "gemini-pro": "gemini-1.0-pro",
            "gemini-pro-latest": "gemini-1.0-pro",
            "gemini-1.5-pro-001": "gemini-1.5-pro",
            "gemini-1.5-flash-001": "gemini-1.5-flash",
        }
        model = remap.get(model, model)
        candidate_models = [model]
        if not model.endswith("-latest"):
            candidate_models.append(f"{model}-latest")

        # ---- 2) Convert messages -> GLM contents ----
        # GLM expects conversation turns with roles "user"|"model".
        # Make sure we include only user/assistant and drop empties.
        ua_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        contents: List[Dict[str, Any]] = []
        for m in ua_msgs:
            text = (m.get("content") or "").strip()
            if not text:
                # Gemini commonly 400s on empty parts; skip them
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": text}]})

        if not contents:
            # Avoid 400 "empty contents"
            raise ProviderError(provider.name, "Gemini request has no non-empty user/assistant messages")

        # Optional system messages become a single systemInstruction Content
        system_texts = [m.get("content", "").strip() for m in messages if m.get("role") == "system"]
        system_texts = [s for s in system_texts if s]
        system_instruction: Optional[Dict[str, Any]] = None
        if system_texts:
            # Spec allows Content without role; omit role to be safe
            system_instruction = {"parts": [{"text": "\n".join(system_texts)}]}

        # ---- 3) Build generationConfig (camelCase) from our params ----
        gc: Dict[str, Any] = {}

        if temperature is not None:
            gc["temperature"] = float(temperature)
        if max_tokens is not None:
            gc["maxOutputTokens"] = int(max_tokens)

        # Accept both snake_case (from your frontend) and camelCase
        p = provider_params or {}
        # snake_case
        if p.get("top_k") is not None:
            gc["topK"] = int(p["top_k"])
        if p.get("top_p") is not None:
            gc["topP"] = float(p["top_p"])
        if p.get("candidate_count") is not None:
            gc["candidateCount"] = int(p["candidate_count"])
        if p.get("stop_sequences") is not None:
            gc["stopSequences"] = list(p["stop_sequences"])
        # camelCase (wins only if not already set)
        if "topK" in p and "topK" not in gc:
            gc["topK"] = int(p["topK"])
        if "topP" in p and "topP" not in gc:
            gc["topP"] = float(p["topP"])
        if "candidateCount" in p and "candidateCount" not in gc:
            gc["candidateCount"] = int(p["candidateCount"])
        if "stopSequences" in p and "stopSequences" not in gc:
            gc["stopSequences"] = list(p["stopSequences"])

        # Sanity guards to avoid 400s on ranges
        if "topP" in gc and not (0.0 <= gc["topP"] <= 1.0):
            del gc["topP"]
        if "topK" in gc and gc["topK"] < 1:
            del gc["topK"]
        if "candidateCount" in gc and not (1 <= gc["candidateCount"] <= 8):
            gc["candidateCount"] = max(1, min(8, int(gc["candidateCount"])))

        payload: Dict[str, Any] = {"contents": contents}
        if gc:
            payload["generationConfig"] = gc
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Optional: safetySettings (array of {category, threshold})
        if "safety_settings" in p:
            payload["safetySettings"] = p["safety_settings"]
        elif "safetySettings" in p:
            payload["safetySettings"] = p["safetySettings"]

        # ---- 4) Call API (v1beta then v1), try model and model-latest ----
        base = provider.base_url.rstrip("/")
        api_key = getattr(provider, "api_key", None)
        params = {"key": api_key} if api_key else {}
        headers = {"Content-Type": "application/json", **(provider.headers or {})}

        endpoints = [
            f"{base}/v1beta/models/{{m}}:generateContent",
            f"{base}/v1/models/{{m}}:generateContent",
        ]

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            last_msg = None
            for ep in endpoints:
                for mname in candidate_models:
                    url = ep.format(m=mname)
                    r = await client.post(url, params=params, headers=headers, json=payload)
                    if r.status_code == 404:
                        # wrong model name for this API version; try next
                        last_msg = r.text
                        continue
                    # Surface error with server JSON if present
                    try:
                        r.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        try:
                            j = r.json()
                            # GLM usually returns {"error": {"message": "..."}}
                            msg = j.get("error", {}).get("message") or j.get("message") or r.text
                        except Exception:
                            msg = r.text
                        raise ProviderError(provider.name, f"{r.status_code} {r.reason_phrase}: {msg}") from e

                    data = r.json() or {}
                    # Typical success path
                    cands = data.get("candidates") or []
                    if not cands:
                        return ""
                    parts = cands[0].get("content", {}).get("parts") or []
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            return part["text"]
                    # if no text parts, return empty (some tool or non-text responses)
                    return ""

        raise ProviderError(provider.name, last_msg or "Gemini request failed (no usable endpoint/model)")



    # ---------------- Ollama ----------------

    async def _ollama_chat(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle Ollama chat completion with optional parameters (omit if None)."""
        url = f"{provider.base_url}/api/chat"

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        if provider_params:
            ollama_options = [
                "mirostat", "mirostat_eta", "mirostat_tau", "num_ctx", "repeat_last_n",
                "repeat_penalty", "tfs_z", "seed", "stop", "top_k", "top_p"
            ]
            for option in ollama_options:
                if option in provider_params:
                    options[option] = provider_params[option]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {}
        }

        if provider_params and "format" in provider_params:
            payload["format"] = provider_params["format"]

        headers = dict(provider.headers or {})

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            self._ensure_ok_stream(r)
            data = r.json()
            return data["message"]["content"]

    # ---------------- Utils ----------------

    async def stream_chat(
            self,
            *,
            provider: Provider,
            model: str,
            messages: list[dict],
            temperature: float | None = None,
            max_tokens: int | None = None,
            min_tokens: int | None = None,
            timeout_s: int = 180,
            provider_params: dict | None = None,
        ) -> AsyncIterator[str]:
            """
            Stream deltas for a single (provider, model). Yield plain text pieces.
            """
            wire = _get_wire(provider)

            if wire == "openai":
                headers = dict(provider.headers or {})
                if getattr(provider, "api_key", None):
                    headers["Authorization"] = f"Bearer {provider.api_key}"
                headers.setdefault("Accept", "text/event-stream")
                headers.setdefault("Content-Type", "application/json")

                url = f"{provider.base_url.rstrip('/')}/chat/completions"

                is_gpt5 = model.startswith(("gpt-5", "o1-", "gpt-4.1"))
                is_gpt4o = model.startswith("gpt-4o")

                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                }

                # tokens
                if max_tokens is not None:
                    if is_gpt5:
                        payload["max_completion_tokens"] = max_tokens
                    else:
                        payload["max_tokens"] = max_tokens

                # temperature
                if temperature is not None:
                    if is_gpt5:
                        payload["temperature"] = 1.0   # gpt-5 ignores/normalizes
                    else:
                        payload["temperature"] = temperature

                # safe provider params
                safe_keys = {"top_p", "frequency_penalty", "presence_penalty", "stop", "tools", "tool_choice", "user"}
                params_src = {}
                if isinstance(provider_params, dict):
                    params_src = provider_params.get("openai_params", provider_params) or {}
                for k, v in (params_src.items() if isinstance(params_src, dict) else []):
                    if v is not None and k in safe_keys:
                        payload[k] = v

                # scrub unsupported extras for gpt-5
                if is_gpt5:
                    payload.pop("response_format", None)
                    payload.pop("seed", None)
                    if "tools" in payload and not payload["tools"]:
                        payload.pop("tools")
                    if "tool_choice" in payload and "tools" not in payload:
                        payload.pop("tool_choice")
                payload = self._sanitize_openai_payload(model, payload)
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as r:
                        
                        async for line in r.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                j = json.loads(data)
                                if "error" in j:
                                    raise ProviderError(provider.name, j["error"].get("message", str(j)))
                                delta = j["choices"][0]["delta"].get("content") or ""
                                if delta:
                                    yield delta
                            except Exception as e:
                                print(f"⚠️ SSE parse error: {e} / {line}")



            if wire == "anthropic":
                # Anthropic streaming via SSE "data:" lines on /v1/messages?stream=true
                headers = dict(provider.headers or {})
                if getattr(provider, "api_key", None):
                    headers["x-api-key"] = provider.api_key
                headers.setdefault("anthropic-version", "2023-06-01")

                url = f"{provider.base_url.rstrip('/')}/v1/messages"

                # Build anthropic payload using your existing helpers
                ap_kwargs: dict = dict(provider_params or {})
                if max_tokens is not None:
                    ap_kwargs["max_tokens"] = max_tokens
                if temperature is not None:
                    ap_kwargs["temperature"] = temperature

                anthropic_params: AnthropicParameters = create_anthropic_params(model=model, **ap_kwargs)

                # Convert messages
                system_texts = [m["content"] for m in messages if m.get("role") == "system"]
                system_str = "\n".join(system_texts) if system_texts else None
                anthropic_messages = []
                for m in messages:
                    role = m.get("role")
                    if role in ("user", "assistant"):
                        anthropic_messages.append({
                            "role": "user" if role == "user" else "assistant",
                            "content": [{"type": "text", "text": m.get("content", "")}],
                        })

                payload = anthropic_params.to_anthropic_payload(messages=anthropic_messages, system=system_str)
                if min_tokens is not None:
                    payload["min_output_tokens"] = int(min_tokens)
                payload["stream"] = True

                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as r:
                        self._ensure_ok_stream(r)
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data = line[6:].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    j = json.loads(data)
                                    t = j.get("type")
                                    if t in ("message_delta", "content_block_delta"):
                                        delta = (j.get("delta") or {}).get("text") or ""
                                        if delta:
                                            yield delta
                                except Exception:
                                    pass
                        return

            if wire == "ollama":
                # Ollama streams JSON objects line-by-line on /api/chat with stream=true
                url = f"{provider.base_url.rstrip('/')}/api/chat"
                options: dict = {}
                if temperature is not None:
                    options["temperature"] = temperature
                if max_tokens is not None:
                    options["num_predict"] = max_tokens
                if provider_params:
                    for k in ("mirostat", "mirostat_eta", "mirostat_tau", "num_ctx", "repeat_penalty", "top_k", "top_p", "stop", "seed"):
                        if k in provider_params:
                            options[k] = provider_params[k]

                payload = {"model": model, "messages": messages, "stream": True, "options": options}

                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    async with client.stream("POST", url, json=payload) as r:
                        self._ensure_ok_stream(r)
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            try:
                                j = json.loads(line)
                                # spec variants: j["message"]["content"] or j["message"]["content"] chunks
                                msg = j.get("message", {})
                                piece = msg.get("content") or j.get("response") or ""
                                if piece:
                                    yield piece
                                if j.get("done"):
                                    break
                            except Exception:
                                pass
                        return

            if wire == "gemini":
                # Simplest: fall back to a single non-streaming chunk. (You can swap to
                # streamGenerateContent later if needed.)
                text = await self._gemini_chat(provider, model, messages, temperature, max_tokens, timeout_s, provider_params)
                if text:
                    yield text
                return

            # Unknown wire => fall back to non-streaming
            text = await self.chat_completion(provider, model, messages, temperature, max_tokens, min_tokens, timeout_s, provider_params)
            if text:
                yield text
    
    def _ensure_ok_stream(self, response: httpx.Response) -> None:
        # Do NOT read response body here — it's a stream.
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code} {response.reason_phrase}")

    
    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise appropriate exception for HTTP errors."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                j = response.json()
                detail = j.get("error", j.get("message", "")) or str(j)
            except Exception:
                detail = response.text
            raise RuntimeError(f"{response.status_code} {response.reason_phrase}: {detail}") from e


# --- Backward compatibility alias (safe removal after full migration) ---
class ChatAdapter(EnhancedChatAdapter):
    pass
