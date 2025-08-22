import asyncio
from typing import Any, Dict, List, Optional
import httpx
import json

from core.exceptions import ProviderError
from providers.base import Provider

# Per-host locks for serialization
_HOST_LOCKS: Dict[str, asyncio.Semaphore] = {}

def get_host_lock(provider: Provider) -> asyncio.Semaphore:
    """Get or create a semaphore for the provider's host."""
    if provider.base_url not in _HOST_LOCKS:
        _HOST_LOCKS[provider.base_url] = asyncio.Semaphore(1)
    return _HOST_LOCKS[provider.base_url]


class ChatAdapter:
    """Adapter for chat completion across different provider types."""
    
    async def chat_completion(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        min_tokens: Optional[int] = None,
        timeout_s: int = 180,
    ) -> str:
        """Generate chat completion using the appropriate provider."""
        try:
            if provider.type == "openai":
                return await self._openai_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s
                )
            elif provider.type == "gemini":
                return await self._gemini_chat(
                    provider, model, messages, temperature, max_tokens, timeout_s
                )
            elif provider.type == "anthropic":
                return await self._anthropic_chat(
                    provider, model, messages, temperature, max_tokens, min_tokens, timeout_s
                )
            else:
                raise ProviderError(provider.name, f"Unsupported provider type: {provider.type}")
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(provider.name, str(e))
    
    async def _openai_chat(
        self, 
        provider: Provider, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int, 
        timeout_s: int
    ) -> str:
        """Handle OpenAI-compatible chat completion."""
        headers = dict(provider.headers or {})
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
            return body
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            use_completion = False
            include_temp = True
            
            # Initial attempt
            r = await client.post(
                url, headers=headers, json=build_payload(use_completion, include_temp)
            )
            
            # Handle parameter compatibility issues
            for _ in range(2):
                if r.status_code != 400:
                    break
                
                try:
                    err = r.json()
                    meta = err.get("error", err)
                    msg = str(meta.get("message", meta))
                except Exception:
                    msg = ""
                
                need_completion = ("max_tokens" in msg and "max_completion_tokens" in msg)
                temp_unsupported = ("temperature" in msg and "supported" in msg)
                
                if not (need_completion or temp_unsupported):
                    break
                
                if need_completion:
                    use_completion = True
                if temp_unsupported:
                    include_temp = False
                
                r = await client.post(
                    url, headers=headers, json=build_payload(use_completion, include_temp)
                )
            
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
        timeout_s: int
    ) -> str:
        """Handle Gemini chat completion."""
        # Convert OpenAI-style messages to Gemini format
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
            payload["systemInstruction"] = {
                "role": "user", 
                "parts": [{"text": "\n".join(sys_texts)}]
            }
        
        headers = dict(provider.headers or {})
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, params=params, headers=headers, json=payload)
            self._raise_for_status(r)
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    
    async def _anthropic_chat(
        self, 
        provider: Provider, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int, 
        min_tokens: Optional[int], 
        timeout_s: int
    ) -> str:
        """Handle Anthropic chat completion."""
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
        
        url = f"{provider.base_url.rstrip('/')}/v1/messages"
        
        # Extract system message
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        system_str = "\n".join(system_texts) if system_texts else None
        
        # Convert remaining messages
        turns = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
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
            self._raise_for_status(r)
            data = r.json()
            
            # Extract text from content blocks
            parts = data.get("content", []) or []
            out: List[str] = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text" and "text" in p:
                    out.append(p["text"])
            return "".join(out) if out else ""
    
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
                pass
            raise RuntimeError(f"{response.status_code} {response.reason_phrase}: {detail}") from e
