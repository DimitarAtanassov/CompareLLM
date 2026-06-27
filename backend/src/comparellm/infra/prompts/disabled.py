"""Null-Object prompt catalog used when Floating Prompts is not configured.

Lists return empty so the UI degrades gracefully; rendering raises a clear
configuration error. This keeps callers free of ``None`` / feature-flag checks.
"""

from __future__ import annotations

from comparellm.errors import ConfigurationError
from comparellm.infra.prompts.base import (
    ProjectRef,
    PromptRef,
    Rendered,
    TagRef,
    VersionRef,
)

_MESSAGE = "Prompt catalog is disabled; set FLOATING_PROMPTS_URL to enable it."


class DisabledPromptCatalog:
    """A catalog that is present but inert."""

    async def list_projects(self) -> list[ProjectRef]:
        return []

    async def list_prompts(self, project: str) -> list[PromptRef]:
        return []

    async def list_versions(self, project: str, name: str) -> list[VersionRef]:
        return []

    async def list_tags(self, project: str, name: str) -> list[TagRef]:
        return []

    async def render(
        self,
        project: str,
        name: str,
        variables: dict[str, object],
        *,
        version: int | None = None,
        tag: str | None = None,
    ) -> Rendered:
        raise ConfigurationError(_MESSAGE)

    async def close(self) -> None:
        return None
