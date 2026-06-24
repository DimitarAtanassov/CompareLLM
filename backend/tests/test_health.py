from __future__ import annotations

from httpx import AsyncClient


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["chat_models"] == 2


async def test_request_id_header(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "x-request-id" in {k.lower() for k in response.headers}
