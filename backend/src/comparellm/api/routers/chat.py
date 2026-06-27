"""Unified chat streaming endpoint.

A single SSE endpoint serves both multi-model compare (many ``targets``) and
single-model interactive chat (one target). It replaces the previous NDJSON
``/chat/stream`` and the LangGraph ``/langgraph/chat/{single,multi}/stream``
endpoints with one well-defined event stream.

Event types: ``start``, ``delta``, ``error``, ``end`` (per model) and ``done``
(terminal). Heartbeat comments keep idle connections alive.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from comparellm.api.deps import ChatServiceDep, ContainerDep
from comparellm.domain.chat_service import ChatEvent
from comparellm.domain.models import ChatMessage
from comparellm.errors import ValidationError
from comparellm.sse import STREAM_HEADERS, sse_comment, sse_event

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatStreamRequest(BaseModel):
    targets: list[str] = Field(..., min_length=1, description="'provider:model' targets")
    messages: list[ChatMessage] = Field(..., min_length=1)
    per_model_params: dict[str, dict[str, object]] = Field(default_factory=dict)
    thread_id: str | None = None


@router.post("/stream")
async def chat_stream(
    body: ChatStreamRequest,
    chat_service: ChatServiceDep,
    container: ContainerDep,
) -> StreamingResponse:
    unknown = [t for t in body.targets if not container.registry.has_chat(t)]
    if unknown:
        raise ValidationError(f"Unknown chat targets: {', '.join(unknown)}")

    heartbeat = container.settings.sse_heartbeat_seconds

    async def event_stream() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for event in chat_service.stream(
                    targets=body.targets,
                    messages=body.messages,
                    per_model_params=body.per_model_params,
                    thread_id=body.thread_id,
                ):
                    await queue.put(event)
            finally:
                await queue.put(None)

        task = asyncio.create_task(pump())
        yield sse_comment("open")
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                except TimeoutError:
                    yield sse_comment("hb")
                    continue
                if event is None:
                    break
                yield sse_event(event.type, event.payload())
        finally:
            task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=STREAM_HEADERS)
