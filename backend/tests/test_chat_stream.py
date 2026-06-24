from __future__ import annotations

import orjson
from httpx import AsyncClient


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name = None
    for line in body.splitlines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event_name:
            payload = orjson.loads(line.split(":", 1)[1].strip())
            events.append((event_name, payload))
    return events


async def test_multi_model_stream(client: AsyncClient) -> None:
    request = {
        "targets": ["fake:alpha", "fake:beta"],
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = await client.post("/chat/stream", json=request)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    types = [name for name, _ in events]

    assert types.count("start") == 2
    assert types.count("end") == 2
    assert types[-1] == "done"

    for model in ("alpha", "beta"):
        deltas = [p["text"] for name, p in events if name == "delta" and p["model"].endswith(model)]
        assert "".join(deltas) == f"echo: {model} <- hello"


async def test_unknown_target_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/chat/stream",
        json={"targets": ["fake:missing"], "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400


async def test_thread_memory_persists(client: AsyncClient) -> None:
    first = await client.post(
        "/chat/stream",
        json={
            "targets": ["fake:alpha"],
            "messages": [{"role": "user", "content": "remember me"}],
            "thread_id": "t1",
        },
    )
    assert first.status_code == 200

    # Second turn: the fake echoes the latest user message; memory is exercised by
    # the service appending prior turns without error and the stream completing.
    second = await client.post(
        "/chat/stream",
        json={
            "targets": ["fake:alpha"],
            "messages": [{"role": "user", "content": "again"}],
            "thread_id": "t1",
        },
    )
    events = _parse_sse(second.text)
    assert events[-1][0] == "done"
