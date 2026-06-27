"""Read-only prompt catalog endpoints.

A thin proxy over the configured prompt store (Floating Prompts). The browser
talks only to CompareLLM; this router calls the store server-to-server. Authoring
happens in the store's own UI, so there are no write endpoints here.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from comparellm.api.deps import PromptCatalogDep
from comparellm.infra.prompts import (
    ProjectRef,
    PromptRef,
    Rendered,
    TagRef,
    VersionRef,
)

router = APIRouter(prefix="/prompts", tags=["prompts"])


class RenderRequest(BaseModel):
    variables: dict[str, object] = Field(default_factory=dict)
    version: int | None = None
    tag: str | None = None


@router.get("/projects")
async def list_projects(catalog: PromptCatalogDep) -> list[ProjectRef]:
    return await catalog.list_projects()


@router.get("/projects/{slug}/prompts")
async def list_prompts(slug: str, catalog: PromptCatalogDep) -> list[PromptRef]:
    return await catalog.list_prompts(slug)


@router.get("/projects/{slug}/prompts/{name}/versions")
async def list_versions(
    slug: str, name: str, catalog: PromptCatalogDep
) -> list[VersionRef]:
    return await catalog.list_versions(slug, name)


@router.get("/projects/{slug}/prompts/{name}/tags")
async def list_tags(slug: str, name: str, catalog: PromptCatalogDep) -> list[TagRef]:
    return await catalog.list_tags(slug, name)


@router.post("/projects/{slug}/prompts/{name}/render")
async def render_prompt(
    slug: str, name: str, body: RenderRequest, catalog: PromptCatalogDep
) -> Rendered:
    return await catalog.render(
        slug, name, body.variables, version=body.version, tag=body.tag
    )
