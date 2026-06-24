"""In-memory session store for development and tests."""

from __future__ import annotations

import asyncio

from app.domain.models import ChatMessage


class MemorySessionStore:
    """Process-local conversation memory."""

    def __init__(self) -> None:
        self._threads: dict[tuple[str, str], list[ChatMessage]] = {}
        self._lock = asyncio.Lock()

    async def get(self, thread_id: str, model: str) -> list[ChatMessage]:
        return list(self._threads.get((thread_id, model), []))

    async def append(self, thread_id: str, model: str, messages: list[ChatMessage]) -> None:
        if not messages:
            return
        async with self._lock:
            self._threads.setdefault((thread_id, model), []).extend(messages)

    async def close(self) -> None:
        self._threads.clear()
