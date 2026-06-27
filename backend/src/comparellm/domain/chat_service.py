"""Chat service: concurrent multi-model streaming with per-thread memory.

Replaces the LangGraph single-/multi-model graphs and the two divergent chat
routers. One code path serves both compare (many targets) and interactive (one
target) chat. Models stream concurrently into a single queue; the API layer
renders the resulting events as SSE.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from comparellm.domain.models import ChatMessage, GenerationParams
from comparellm.infra.session.base import SessionStore
from comparellm.log import get_logger
from comparellm.providers.registry import ProviderRegistry

log = get_logger(__name__)


@dataclass(frozen=True)
class ChatEvent:
    """A single streaming event emitted by :meth:`ChatService.stream`."""

    type: str  # "start" | "delta" | "error" | "end" | "done"
    model: str | None = None
    text: str | None = None
    error: str | None = None

    def payload(self) -> dict[str, Any]:
        # The event type is included in the data payload (in addition to the SSE
        # ``event:`` name) so consumers can dispatch on a single self-describing object.
        data: dict[str, Any] = {"type": self.type}
        if self.model is not None:
            data["model"] = self.model
        if self.text is not None:
            data["text"] = self.text
        if self.error is not None:
            data["error"] = self.error
        return data


class ChatService:
    """Fans a conversation out across one or more chat models."""

    def __init__(self, registry: ProviderRegistry, sessions: SessionStore) -> None:
        self._registry = registry
        self._sessions = sessions

    async def stream(
        self,
        *,
        targets: list[str],
        messages: list[ChatMessage],
        per_model_params: dict[str, dict[str, Any]],
        thread_id: str | None,
    ) -> AsyncIterator[ChatEvent]:
        """Yield streaming events for every target until all complete.

        When ``thread_id`` is set, prior conversation for each ``(thread_id,
        target)`` is prepended and the new turn + response are persisted, giving
        shared memory across compare and interactive views. System messages are
        treated as ephemeral per-request context and never persisted.
        """
        system_messages = [m for m in messages if m.role == "system"]
        new_turn = [m for m in messages if m.role != "system"]
        queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()

        async def run_target(target: str) -> None:
            await queue.put(ChatEvent("start", model=target))
            collected: list[str] = []
            try:
                provider = self._registry.get_chat(target)
                params = GenerationParams.model_validate(per_model_params.get(target, {}))
                history = await self._sessions.get(thread_id, target) if thread_id else []
                conversation = system_messages + history + new_turn

                async for delta in provider.stream(conversation, params):
                    collected.append(delta)
                    await queue.put(ChatEvent("delta", model=target, text=delta))

                if thread_id:
                    await self._persist(thread_id, target, new_turn, "".join(collected))
            except Exception as exc:  # noqa: BLE001 - surface per-model, keep others alive
                log.warning("chat_target_failed", target=target, error=str(exc))
                await queue.put(ChatEvent("error", model=target, error=str(exc)))
            finally:
                await queue.put(ChatEvent("end", model=target))

        async def supervise() -> None:
            await asyncio.gather(*(run_target(t) for t in targets), return_exceptions=True)
            await queue.put(None)

        supervisor = asyncio.create_task(supervise())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            yield ChatEvent("done")
        finally:
            supervisor.cancel()

    async def _persist(
        self, thread_id: str, target: str, new_turn: list[ChatMessage], answer: str
    ) -> None:
        to_store = list(new_turn)
        if answer:
            to_store.append(ChatMessage(role="assistant", content=answer))
        await self._sessions.append(thread_id, target, to_store)
