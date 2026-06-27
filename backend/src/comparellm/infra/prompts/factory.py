"""Factory that selects the prompt catalog backend from settings."""

from __future__ import annotations

from comparellm.infra.prompts.base import PromptCatalog
from comparellm.settings import Settings


def build_prompt_catalog(settings: Settings) -> PromptCatalog:
    """Return an HTTP catalog when ``FLOATING_PROMPTS_URL`` is set, else disabled."""
    if settings.floating_prompts_url:
        from comparellm.infra.prompts.http import HttpPromptCatalog

        return HttpPromptCatalog(settings.floating_prompts_url)

    from comparellm.infra.prompts.disabled import DisabledPromptCatalog

    return DisabledPromptCatalog()
