"""HTTP-backed prompt catalog: a thin client over the Floating Prompts API.

Server-to-server only (the browser never calls Floating Prompts directly), so no
auth or CORS is involved. Upstream failures are mapped to CompareLLM's error
hierarchy so the API surface stays consistent.
"""

from __future__ import annotations

from typing import Any

import httpx

from comparellm.errors import NotFoundError, UpstreamError, ValidationError
from comparellm.infra.prompts.base import (
    ProjectRef,
    PromptRef,
    Rendered,
    TagRef,
    VersionRef,
)

_API = "/api/v1"


class HttpPromptCatalog:
    """Read-only Floating Prompts client."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def _get(self, path: str, **kwargs: Any) -> Any:
        return self._unwrap(await self._request("GET", path, **kwargs))

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:  # transport-level failure
            raise UpstreamError(f"Floating Prompts unreachable: {exc}") from exc

    @staticmethod
    def _unwrap(response: httpx.Response) -> Any:
        if response.is_success:
            return response.json()
        detail = _detail(response)
        if response.status_code == 404:
            raise NotFoundError(detail)
        if response.status_code in (400, 422):
            raise ValidationError(detail)
        raise UpstreamError(detail)

    async def list_projects(self) -> list[ProjectRef]:
        data = await self._get(f"{_API}/projects", params={"limit": 200})
        return [ProjectRef.model_validate(p) for p in data["items"]]

    async def list_prompts(self, project: str) -> list[PromptRef]:
        data = await self._get(f"{_API}/projects/{project}/prompts", params={"limit": 200})
        return [PromptRef.model_validate(p) for p in data["items"]]

    async def list_versions(self, project: str, name: str) -> list[VersionRef]:
        data = await self._get(f"{_API}/projects/{project}/prompts/{name}/versions")
        return [VersionRef.model_validate(v) for v in data]

    async def list_tags(self, project: str, name: str) -> list[TagRef]:
        data = await self._get(f"{_API}/projects/{project}/prompts/{name}/tags")
        return [TagRef.model_validate(t) for t in data]

    async def render(
        self,
        project: str,
        name: str,
        variables: dict[str, object],
        *,
        version: int | None = None,
        tag: str | None = None,
    ) -> Rendered:
        body = {"variables": variables, "version": version, "tag": tag}
        resp = await self._request(
            "POST", f"{_API}/projects/{project}/prompts/{name}/render", json=body
        )
        return Rendered.model_validate(self._unwrap(resp))

    async def close(self) -> None:
        await self._client.aclose()


def _detail(response: httpx.Response) -> str:
    """Best-effort human-readable detail from a Floating Prompts error response."""
    try:
        body = response.json()
    except ValueError:
        return f"Floating Prompts error ({response.status_code})"
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return f"Floating Prompts error ({response.status_code})"
