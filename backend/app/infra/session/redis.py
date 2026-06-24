"""Redis-backed session store for production deployments.

Selected via ``SESSION_BACKEND=redis`` and ``REDIS_URL``. Each ``(thread_id,
model)`` conversation is stored as a Redis list of JSON-encoded messages with a
configurable TTL so idle sessions expire automatically.
"""

from __future__ import annotations

import orjson
import redis.asyncio as redis

from app.domain.models import ChatMessage


class RedisSessionStore:
    """Redis list-backed conversation memory."""

    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self._client: redis.Redis = redis.from_url(redis_url, decode_responses=False)
        self._ttl = ttl_seconds

    @staticmethod
    def _key(thread_id: str, model: str) -> str:
        return f"chat:{thread_id}:{model}"

    async def get(self, thread_id: str, model: str) -> list[ChatMessage]:
        raw = await self._client.lrange(self._key(thread_id, model), 0, -1)
        return [ChatMessage.model_validate(orjson.loads(item)) for item in raw]

    async def append(self, thread_id: str, model: str, messages: list[ChatMessage]) -> None:
        if not messages:
            return
        key = self._key(thread_id, model)
        payloads = [orjson.dumps(message.model_dump()) for message in messages]
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *payloads)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def close(self) -> None:
        await self._client.aclose()
