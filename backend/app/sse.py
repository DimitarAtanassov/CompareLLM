"""Server-Sent Events (SSE) encoding helpers.

A single, well-defined SSE wire format is used by the chat streaming endpoint,
replacing the previous mixture of NDJSON and ad-hoc SSE shapes. Every event is a
named event (``event:`` line) carrying a JSON ``data:`` payload.
"""

from __future__ import annotations

from typing import Any

import orjson

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable proxy buffering so deltas reach the client immediately.
    "X-Accel-Buffering": "no",
}


def sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Encode a named SSE event with a JSON payload."""
    payload = orjson.dumps(data).decode("utf-8")
    return f"event: {event}\ndata: {payload}\n\n".encode()


def sse_comment(text: str = "") -> bytes:
    """Encode an SSE comment line, used for heartbeats."""
    return f": {text}\n\n".encode()
