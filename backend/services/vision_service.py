from __future__ import annotations
import asyncio
import base64
import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from core.model_factory import resolve_and_init_from_registry
from graphs.factory import build_single_model_graph, build_multi_model_graph

class VisionService:
    STREAM_HEADERS = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    def _is_png_or_jpeg(self, mime: str) -> bool:
        m = (mime or "").lower()
        return m in ("image/png", "image/jpeg", "image/jpg")

    def _b64(self, s: bytes) -> str:
        return base64.b64encode(s).decode("ascii")

    def _b64_data_uri(self, mime: str, b: bytes) -> str:
        return f"data:{mime};base64,{self._b64(b)}"

    def _anthropic_part(self, mime: str, b: bytes) -> dict:
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": self._b64(b)}}

    def _image_url_part(self, data_uri: str) -> dict:
        return {"type": "image_url", "image_url": {"url": data_uri}}

    def _system_msg_if(self, system_text: Optional[str]) -> List[BaseMessage]:
        return [SystemMessage(content=system_text)] if system_text else []

    def _build_vision_human_message(self, provider_type: str, prompt_text: Optional[str], mime: str, raw: bytes) -> HumanMessage:
        ptype = (provider_type or "").lower()
        # Cerebras expects a plain string, not a list of parts
        if ptype == "cerebras":
            return HumanMessage(content=prompt_text or "")
        parts: List[dict] = []
        if prompt_text:
            parts.append({"type": "text", "text": prompt_text})
        if ptype == "anthropic":
            parts.append(self._anthropic_part(mime, raw))
        else:
            parts.append(self._image_url_part(self._b64_data_uri(mime, raw)))
        return HumanMessage(content=parts)

    def _extract_text_from_output(self, output: Any) -> str:
        if isinstance(output, AIMessage):
            c = output.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                buf: List[str] = []
                for part in c:
                    if isinstance(part, dict):
                        if part.get("type") in ("text", "raw"):
                            t = part.get("text") or part.get("value") or ""
                            if isinstance(t, str):
                                buf.append(t)
                if buf:
                    return "".join(buf)
            t = getattr(output, "text", None)
            if isinstance(t, str):
                return t
            return ""
        if isinstance(output, dict):
            for k in ("text", "message", "content"):
                v = output.get(k)
                if isinstance(v, str):
                    return v
        if isinstance(output, str):
            return output
        if hasattr(output, "content") and isinstance(getattr(output, "content"), str):
            return getattr(output, "content")
        return ""

    def _extract_image_from_output(self, output: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        parts: List[Any] = []
        if isinstance(output, AIMessage):
            if isinstance(output.content, list):
                parts = output.content
        elif isinstance(output, dict) and isinstance(output.get("content"), list):
            parts = output["content"]
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if isinstance(url, str) and url:
                    return None, None, url
            if part.get("type") in ("image", "image_base64"):
                src = part.get("source") or {}
                data = src.get("data") or part.get("data")
                mime = src.get("media_type") or part.get("mime") or "image/png"
                if isinstance(data, str) and data:
                    if data.startswith("data:"):
                        try:
                            header, b64 = data.split(",", 1)
                            mime_guess = header.split(";")[0].split(":", 1)[1] if ":" in header else "image/png"
                            return b64, mime_guess, data
                        except Exception:
                            pass
                    return data, mime, None
        if isinstance(output, AIMessage):
            ak = getattr(output, "additional_kwargs", {}) or {}
            for key in ("image_base64", "image_b64", "image"):
                val = ak.get(key)
                if isinstance(val, str) and val:
                    return val, ak.get("image_mime") or "image/png", None
        return None, None, None

    def _json_or_none(self, s: Optional[str]) -> Optional[Union[dict, list]]:
        if not s:
            return None
        try:
            val = json.loads(s)
            if isinstance(val, (dict, list)):
                return val
        except Exception:
            pass
        return None

    def _is_overload_error(self, err: Exception) -> bool:
        msg = (str(err) or "").lower()
        if "overloaded" in msg or "status 529" in msg or "code: 529" in msg:
            return True
        status = getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)
        if status == 529:
            return True
        return False

    def _backoff_delay(self, attempt: int, base: float = 0.6, max_backoff: float = 6.0) -> float:
        low = base * (2 ** attempt)
        high = base * (2 ** (attempt + 1))
        return min(random.uniform(low, high), max_backoff)

    async def ainvoke_with_retries(self, llm, msgs, *, max_retries: int = 3, base_delay: float = 0.6):
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await llm.ainvoke(msgs)
            except Exception as e:
                if self._is_overload_error(e) and attempt < max_retries:
                    delay = self._backoff_delay(attempt, base=base_delay)
                    await asyncio.sleep(delay)
                    continue
                last_err = e
                break
        if last_err and self._is_overload_error(last_err):
            raise HTTPException(status_code=503, detail="Provider is overloaded. Please retry shortly.")
        if last_err:
            raise HTTPException(status_code=502, detail=str(last_err))
        raise HTTPException(status_code=500, detail="Unknown error during model invocation")

    def validate_upload(self, up: UploadFile) -> None:
        if not up.content_type or not self._is_png_or_jpeg(up.content_type):
            raise HTTPException(400, "Only image/png and image/jpeg are supported")

    async def read_upload(self, up: UploadFile) -> bytes:
        raw = await up.read()
        if not raw:
            raise HTTPException(400, "Empty image upload")
        return raw

    def choose_upload(self, image: Optional[UploadFile], file_: Optional[UploadFile]) -> UploadFile:
        if file_ is not None:
            return file_
        if image is not None:
            return image
        raise HTTPException(400, "Missing file upload: expected form field 'file'")

    def provider_type_from_wire(self, reg, wire: str) -> Tuple[str, str, str]:
        try:
            pkey, model_name = wire.split(":", 1)
        except ValueError:
            raise HTTPException(400, "wire must be 'provider:model'")
        ptype = (reg.provider_type(pkey) or "").lower()
        return pkey, model_name, ptype

    def get_registry(self, request) -> Any:
        reg = getattr(request.app.state, "registry", None)
        if reg is None:
            raise HTTPException(500, "Model registry not initialized")
        return reg

    def get_memory_backend(self, request) -> Any:
        return getattr(request.app.state, "graph_memory", None)

    def resolve_and_init_model(self, reg, wire: str, params: dict) -> Any:
        return resolve_and_init_from_registry(registry=reg, wire=wire, params=params)

    # Add more service methods as needed for analyze, single_stream, multi_stream
