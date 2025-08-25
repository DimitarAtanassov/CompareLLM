import asyncio
from typing import Any, Dict, List, Optional
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
            print(f"🔑 API Key present: {bool(provider.api_key)}")
            print(f"🌐 Base URL: {provider.base_url}")
            print(f"📝 Messages count: {len(messages)}")
            if provider_params:
                print(f"⚙️  Provider params: {json.dumps(provider_params, indent=2)}")

            if provider.type == "openai":
                # This path also works for any OpenAI-compatible provider when configured as type "openai".
                return await self._openai_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s, provider_params
                )
            elif provider.type == "deepseek":
                # Explicit DeepSeek path if you register it as its own type.
                return await self._deepseek_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s, provider_params
                )
            elif provider.type == "gemini":
                return await self._gemini_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s, provider_params
                )
            elif provider.type == "anthropic":
                return await self._anthropic_chat_enhanced(
                    provider, model, messages, temperature, max_tokens, min_tokens, timeout_s, provider_params
                )
            elif provider.type == "ollama":
                return await self._ollama_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s, provider_params
                )
            else:
                raise ProviderError(provider.name, f"Unsupported provider type: {provider.type}")

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
        if provider.api_key:
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
                self._raise_for_status(r)
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

    # ---------------- OpenAI ----------------

    async def _openai_chat(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle OpenAI-compatible chat completion with optional parameters (omit if None)."""
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        url = f"{provider.base_url}/chat/completions"

        # Only adjust temperature for GPT-5 if caller provided one
        adjusted_temperature = temperature
        if model == "gpt-5" and adjusted_temperature is not None and adjusted_temperature != 1.0:
            print(f"⚠️  Adjusting temperature for {model}: {adjusted_temperature} -> 1.0 (model constraint)")
            adjusted_temperature = 1.0

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if adjusted_temperature is not None:
            payload["temperature"] = adjusted_temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add provider-specific params if present
        if provider_params:
            openai_params = [
                "top_p", "frequency_penalty", "presence_penalty", "stop",
                "stream", "logit_bias", "user", "functions", "function_call",
                "tools", "tool_choice", "response_format", "seed"
            ]
            for param in openai_params:
                if param in provider_params:
                    payload[param] = provider_params[param]

        print(f"📤 OpenAI payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)

            # Retry for known param-name edge-cases ONLY if we sent those keys
            if r.status_code == 400:
                try:
                    err = r.json()
                    msg = str(err.get("error", {}).get("message", ""))

                    if "max_tokens" in payload and "max_tokens" in msg and "max_completion_tokens" in msg:
                        payload["max_completion_tokens"] = payload.pop("max_tokens")
                        print("🔄 Retrying with max_completion_tokens")
                        r = await client.post(url, headers=headers, json=payload)

                    elif "temperature" in payload and "temperature" in msg and "supported" in msg and model == "gpt-5":
                        print(f"🔄 Removing temperature parameter for {model} (using default)")
                        del payload["temperature"]
                        r = await client.post(url, headers=headers, json=payload)

                except Exception:
                    pass

            self._raise_for_status(r)
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
        """Handle Gemini chat completion with optional parameters (omit if None)."""
        sys_texts = [m["content"] for m in messages if m.get("role") == "system"]
        ua_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]

        contents = []
        for m in ua_msgs:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        params = {"key": provider.api_key} if provider.api_key else {}
        url = f"{provider.base_url}/v1beta/models/{model}:generateContent"

        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        # Merge Gemini-specific params without overwriting unset fields
        if provider_params:
            # accept common camelCase keys as used by Google
            if "topK" in provider_params:
                generation_config["topK"] = provider_params["topK"]
            if "topP" in provider_params:
                generation_config["topP"] = provider_params["topP"]
            if "candidateCount" in provider_params:
                generation_config["candidateCount"] = provider_params["candidateCount"]
            if "stopSequences" in provider_params:
                generation_config["stopSequences"] = provider_params["stopSequences"]
            # allow passing temperature/maxOutputTokens from provider_params too, but
            # only if we didn't explicitly set them above
            if "temperature" in provider_params and "temperature" not in generation_config:
                generation_config["temperature"] = provider_params["temperature"]
            if "maxOutputTokens" in provider_params and "maxOutputTokens" not in generation_config:
                generation_config["maxOutputTokens"] = provider_params["maxOutputTokens"]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config or {},
        }

        if sys_texts:
            payload["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": "\n".join(sys_texts)}]
            }

        if provider_params and "safetySettings" in provider_params:
            payload["safetySettings"] = provider_params["safetySettings"]

        headers = dict(provider.headers or {})

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, params=params, headers=headers, json=payload)
            self._raise_for_status(r)
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

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
            self._raise_for_status(r)
            data = r.json()
            return data["message"]["content"]

    # ---------------- DeepSeek (OpenAI-compatible) ----------------

    async def _deepseek_chat(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle DeepSeek chat completion (OpenAI-compatible)."""
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        url = f"{provider.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if provider_params:
            supported_params = [
                "top_p", "frequency_penalty", "presence_penalty", "stop",
                "stream", "n", "logprobs", "echo", "seed"
            ]
            for param in supported_params:
                if param in provider_params:
                    payload[param] = provider_params[param]

        print(f"🚀 DeepSeek request - Model: {model}")
        print(f"📝 Payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            try:
                r = await client.post(url, headers=headers, json=payload)
                self._raise_for_status(r)
                data = r.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                raise ProviderError(provider.name, "Invalid response format from DeepSeek")

            except httpx.HTTPStatusError as e:
                error_detail = ""
                try:
                    error_json = r.json()
                    error_detail = error_json.get("error", {}).get("message", r.text)
                except Exception:
                    error_detail = r.text
                print(f"❌ DeepSeek HTTP error: {e.response.status_code} - {error_detail}")
                raise ProviderError(provider.name, f"HTTP {e.response.status_code}: {error_detail}")

    # ---------------- Utils ----------------

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
