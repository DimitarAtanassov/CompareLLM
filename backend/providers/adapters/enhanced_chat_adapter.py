import asyncio
from typing import Any, Dict, List, Optional, Union
import httpx
import json
import os

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
        temperature: float = 0.7,
        max_tokens: int = 8192,
        min_tokens: Optional[int] = None,
        timeout_s: int = 180,
        provider_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate chat completion using the appropriate provider with enhanced parameters.
        
        Args:
            provider: Provider configuration
            model: Model name
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            min_tokens: Minimum tokens to generate (Anthropic only)
            timeout_s: Request timeout
            provider_params: Provider-specific parameters
        """
        try:
            print(f"🚀 Enhanced Chat request - Provider: {provider.name}, Model: {model}")
            print(f"🔑 API Key present: {bool(provider.api_key)}")
            print(f"🌐 Base URL: {provider.base_url}")
            print(f"📝 Messages count: {len(messages)}")
            
            if provider_params:
                print(f"⚙️  Provider params: {json.dumps(provider_params, indent=2)}")
            
            if provider.type == "openai":
                return await self._openai_chat(
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
    
    async def _anthropic_chat_enhanced(
        self, 
        provider: Provider, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int, 
        min_tokens: Optional[int], 
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle Anthropic chat completion with full parameter support."""
        
        # Create Anthropic parameters
        anthropic_params = create_anthropic_params(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **(provider_params or {})
        )
        
        print(f"🤖 Using Anthropic parameters: {anthropic_params.model_dump(exclude_unset=True)}")
        
        # Check model capabilities
        model_config = get_model_config(model)
        if model_config:
            if anthropic_params.thinking and not model_config.supports_thinking:
                print(f"⚠️  Warning: Model {model} may not support extended thinking")
            print(f"📊 Model info: Context={model_config.max_context_tokens}, RPM={model_config.default_rpm}")
        
        # Set up headers
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
        
        url = f"{provider.base_url.rstrip('/')}/v1/messages"
        
        # Extract system message
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        system_str = "\n".join(system_texts) if system_texts else None
        
        # Convert remaining messages to Anthropic format
        anthropic_messages = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            anthropic_messages.append({
                "role": "user" if role == "user" else "assistant",
                "content": [{"type": "text", "text": m.get("content", "")}],
            })
        
        # Add min_tokens if specified (Anthropic-specific)
        extra_params = {}
        if min_tokens is not None:
            extra_params["min_output_tokens"] = int(min_tokens)
        
        # Create the payload using our parameter class
        try:
            payload = anthropic_params.to_anthropic_payload(
                messages=anthropic_messages,
                system=system_str
            )
            
            # Add any extra parameters
            payload.update(extra_params)
            
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
                
                # Log extended thinking if present
                if "content" in data:
                    for content_block in data["content"]:
                        if content_block.get("type") == "thinking":
                            thinking_text = content_block.get("thinking", "")
                            print(f"🧠 Extended thinking: {thinking_text[:100]}..." if len(thinking_text) > 100 else f"🧠 Extended thinking: {thinking_text}")
                
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
                except:
                    error_detail = r.text
                
                print(f"❌ Anthropic HTTP error: {e.response.status_code} - {error_detail}")
                raise ProviderError(provider.name, f"HTTP {e.response.status_code}: {error_detail}")
    
    async def _openai_chat(
        self, 
        provider: Provider, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int, 
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle OpenAI-compatible chat completion with optional parameters."""
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        
        url = f"{provider.base_url}/chat/completions"
        
        # Handle model-specific constraints
        adjusted_temperature = temperature
        if model == "gpt-5":
            # GPT-5 only supports temperature = 1.0
            if temperature != 1.0:
                print(f"⚠️  Adjusting temperature for {model}: {temperature} -> 1.0 (model constraint)")
                adjusted_temperature = 1.0
        
        # Build base payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": adjusted_temperature,
            "max_tokens": max_tokens,
        }
        
        # Add provider-specific parameters
        if provider_params:
            # Common OpenAI parameters
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
            # Handle parameter compatibility (existing logic)
            r = await client.post(url, headers=headers, json=payload)
            
            # Retry logic for max_completion_tokens vs max_tokens and temperature issues
            if r.status_code == 400:
                try:
                    err = r.json()
                    msg = str(err.get("error", {}).get("message", ""))
                    
                    # Handle max_tokens parameter name
                    if "max_tokens" in msg and "max_completion_tokens" in msg:
                        payload["max_completion_tokens"] = payload.pop("max_tokens")
                        print("🔄 Retrying with max_completion_tokens")
                        r = await client.post(url, headers=headers, json=payload)
                    
                    # Handle temperature constraint for GPT-5
                    elif "temperature" in msg and "supported" in msg and model == "gpt-5":
                        if "temperature" in payload:
                            print(f"🔄 Removing temperature parameter for {model} (using default)")
                            del payload["temperature"]
                            r = await client.post(url, headers=headers, json=payload)
                    
                except:
                    pass
            
            self._raise_for_status(r)
            data = r.json()
            return data["choices"][0]["message"]["content"]
    
    async def _gemini_chat(
        self, 
        provider: Provider, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int, 
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle Gemini chat completion with optional parameters."""
        # Convert OpenAI-style messages to Gemini format
        sys_texts = [m["content"] for m in messages if m["role"] == "system"]
        user_assistant_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
        
        contents = []
        for m in user_assistant_msgs:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        
        params = {"key": provider.api_key} if provider.api_key else {}
        url = f"{provider.base_url}/v1beta/models/{model}:generateContent"
        
        generation_config = {
            "temperature": temperature, 
            "maxOutputTokens": max_tokens
        }
        
        # Add Gemini-specific parameters
        if provider_params:
            gemini_params = [
                "topK", "topP", "candidateCount", "stopSequences", 
                "maxOutputTokens", "temperature"
            ]
            for param in gemini_params:
                if param in provider_params:
                    generation_config[param] = provider_params[param]
        
        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        
        if sys_texts:
            payload["systemInstruction"] = {
                "role": "user", 
                "parts": [{"text": "\n".join(sys_texts)}]
            }
        
        # Add safety settings if provided
        if provider_params and "safetySettings" in provider_params:
            payload["safetySettings"] = provider_params["safetySettings"]
        
        headers = dict(provider.headers or {})
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, params=params, headers=headers, json=payload)
            self._raise_for_status(r)
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    
    async def _ollama_chat(
        self, 
        provider: Provider, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int, 
        timeout_s: int,
        provider_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle Ollama chat completion with optional parameters."""
        url = f"{provider.base_url}/api/chat"
        
        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }
        
        # Add Ollama-specific options
        if provider_params:
            ollama_options = [
                "mirostat", "mirostat_eta", "mirostat_tau", "num_ctx", "repeat_last_n",
                "repeat_penalty", "tfs_z", "seed", "stop", "top_k", "top_p"
            ]
            for option in ollama_options:
                if option in provider_params:
                    options[option] = provider_params[option]
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        # Add format if specified
        if provider_params and "format" in provider_params:
            payload["format"] = provider_params["format"]
        
        headers = dict(provider.headers or {})
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            self._raise_for_status(r)
            data = r.json()
            return data["message"]["content"]
    
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