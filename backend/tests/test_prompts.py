"""Tests for the prompt catalog client and the /prompts router."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from comparellm.errors import NotFoundError, UpstreamError
from comparellm.infra.prompts.base import (
    ProjectRef,
    PromptRef,
    Rendered,
    TagRef,
    VersionRef,
)
from comparellm.infra.prompts.http import HttpPromptCatalog
from comparellm.main import create_app
from tests.conftest import FakeContainer


def _catalog(handler: httpx.MockTransport) -> HttpPromptCatalog:
    return HttpPromptCatalog(
        "http://fp",
        client=httpx.AsyncClient(base_url="http://fp", transport=handler),
    )


# -- HttpPromptCatalog (MockTransport) --------------------------------------


async def test_list_projects_parses_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/projects"
        return httpx.Response(
            200,
            json={
                "items": [{"slug": "acme", "name": "ACME", "description": None}],
                "meta": {"total": 1, "limit": 200, "offset": 0},
            },
        )

    catalog = _catalog(httpx.MockTransport(handler))
    projects = await catalog.list_projects()
    assert projects == [ProjectRef(slug="acme", name="ACME")]
    await catalog.close()


async def test_render_posts_body_and_parses() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        assert request.method == "POST"
        assert request.url.path == "/api/v1/projects/acme/prompts/greeter/render"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "name": "greeter",
                "version": 2,
                "system_prompt": "Be nice",
                "user_prompt": "Hi Ada",
            },
        )

    catalog = _catalog(httpx.MockTransport(handler))
    result = await catalog.render("acme", "greeter", {"name": "Ada"}, tag="production")
    assert captured == {"variables": {"name": "Ada"}, "version": None, "tag": "production"}
    assert result == Rendered(
        name="greeter", version=2, system_prompt="Be nice", user_prompt="Hi Ada"
    )
    await catalog.close()


async def test_404_maps_to_not_found() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Prompt 'x' not found.", "code": "not_found"})

    catalog = _catalog(httpx.MockTransport(handler))
    with pytest.raises(NotFoundError, match="not found"):
        await catalog.list_versions("acme", "x")
    await catalog.close()


async def test_transport_error_maps_to_upstream() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    catalog = _catalog(httpx.MockTransport(handler))
    with pytest.raises(UpstreamError, match="unreachable"):
        await catalog.list_projects()
    await catalog.close()


# -- /prompts router --------------------------------------------------------


class StubCatalog:
    """Canned catalog for router tests."""

    async def list_projects(self) -> list[ProjectRef]:
        return [ProjectRef(slug="acme", name="ACME")]

    async def list_prompts(self, project: str) -> list[PromptRef]:
        return [PromptRef(name="greeter")]

    async def list_versions(self, project: str, name: str) -> list[VersionRef]:
        return [VersionRef(version=1, user_prompt="Hi {{ name }}")]

    async def list_tags(self, project: str, name: str) -> list[TagRef]:
        return [TagRef(name="production")]

    async def render(
        self,
        project: str,
        name: str,
        variables: dict[str, object],
        *,
        version: int | None = None,
        tag: str | None = None,
    ) -> Rendered:
        return Rendered(
            name=name, version=1, system_prompt=None, user_prompt=f"Hi {variables.get('name')}"
        )

    async def close(self) -> None:
        return None


@pytest.fixture
async def stub_client() -> AsyncIterator[AsyncClient]:
    container = FakeContainer()
    container.prompt_catalog = StubCatalog()  # type: ignore[assignment]
    app = create_app(container.settings)
    app.state.container = container
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_router_lists_projects(stub_client: AsyncClient) -> None:
    resp = await stub_client.get("/prompts/projects")
    assert resp.status_code == 200
    assert resp.json() == [{"slug": "acme", "name": "ACME", "description": None}]


async def test_router_renders(stub_client: AsyncClient) -> None:
    resp = await stub_client.post(
        "/prompts/projects/acme/prompts/greeter/render",
        json={"variables": {"name": "Ada"}},
    )
    assert resp.status_code == 200
    assert resp.json()["user_prompt"] == "Hi Ada"


async def test_disabled_catalog_returns_empty(client: AsyncClient) -> None:
    # Default FakeContainer uses DisabledPromptCatalog -> graceful empty list.
    resp = await client.get("/prompts/projects")
    assert resp.status_code == 200
    assert resp.json() == []
