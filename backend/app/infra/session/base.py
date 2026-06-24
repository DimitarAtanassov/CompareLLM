"""Session store protocol for per-thread, per-model conversation memory.

Memory is keyed by ``(thread_id, model)`` so each compared model keeps a coherent
conversation while a single ``thread_id`` ties the compare and interactive views
together. This replaces LangGraph's ``InMemorySaver`` with a horizontally
scalable, pluggable backend.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.models import ChatMessage


class SessionStore(Protocol):
    """Durable-ish conversation memory."""

    async def get(self, thread_id: str, model: str) -> list[ChatMessage]:
        """Return the stored conversation for a thread/model (empty if none)."""
        ...

    async def append(
        self, thread_id: str, model: str, messages: list[ChatMessage]
    ) -> None:
        """Append messages to the stored conversation."""
        ...

    async def close(self) -> None:
        """Release any held resources."""
        ...
