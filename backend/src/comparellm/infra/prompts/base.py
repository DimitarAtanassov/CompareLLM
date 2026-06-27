"""Prompt catalog protocol and DTOs.

A read-only view over an external prompt store (Floating Prompts). CompareLLM
loads prompts here to use as system prompts; all authoring stays in that store's
own UI. The protocol keeps the rest of the app decoupled from the transport and
lets us swap in a disabled (Null-Object) implementation when the feature is off.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class ProjectRef(BaseModel):
    """A project (namespace) in the prompt store."""

    slug: str
    name: str
    description: str | None = None


class PromptRef(BaseModel):
    """A prompt identity within a project."""

    name: str
    description: str | None = None


class VariableRef(BaseModel):
    """A declared template variable of a prompt version."""

    name: str
    required: bool = True
    description: str | None = None


class VersionRef(BaseModel):
    """One immutable version of a prompt."""

    version: int
    system_prompt: str | None = None
    user_prompt: str
    variables: list[VariableRef] = []


class TagRef(BaseModel):
    """A movable alias pointing at a version."""

    name: str
    version: int | None = None


class Rendered(BaseModel):
    """The result of rendering a prompt version with variable values."""

    name: str
    version: int
    system_prompt: str | None = None
    user_prompt: str


class PromptCatalog(Protocol):
    """Read-only access to the external prompt store."""

    async def list_projects(self) -> list[ProjectRef]:
        """Return all projects (empty when the feature is disabled)."""
        ...

    async def list_prompts(self, project: str) -> list[PromptRef]:
        """Return the prompts in a project."""
        ...

    async def list_versions(self, project: str, name: str) -> list[VersionRef]:
        """Return all versions of a prompt, newest first."""
        ...

    async def list_tags(self, project: str, name: str) -> list[TagRef]:
        """Return the tags (aliases) of a prompt."""
        ...

    async def render(
        self,
        project: str,
        name: str,
        variables: dict[str, object],
        *,
        version: int | None = None,
        tag: str | None = None,
    ) -> Rendered:
        """Render a prompt version with variable values."""
        ...

    async def close(self) -> None:
        """Release any held resources."""
        ...
