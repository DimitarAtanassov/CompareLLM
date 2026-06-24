from __future__ import annotations

from app.domain.models import GenerationParams
from app.providers.params import (
    anthropic_kwargs,
    cohere_kwargs,
    gemini_config_kwargs,
    openai_kwargs,
)


def test_openai_kwargs_omits_none() -> None:
    params = GenerationParams(temperature=0.7, max_tokens=256)
    out = openai_kwargs(params)
    assert out == {"temperature": 0.7, "max_tokens": 256}
    assert "top_p" not in out


def test_anthropic_defaults_max_tokens() -> None:
    out = anthropic_kwargs(GenerationParams(), default_max_tokens=1024)
    assert out["max_tokens"] == 1024


def test_anthropic_thinking_disables_temperature() -> None:
    out = anthropic_kwargs(
        GenerationParams(temperature=0.9, thinking_budget=2000), default_max_tokens=1024
    )
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert "temperature" not in out


def test_gemini_renames_max_tokens() -> None:
    out = gemini_config_kwargs(GenerationParams(max_tokens=128, top_k=40))
    assert out["max_output_tokens"] == 128
    assert out["top_k"] == 40


def test_cohere_renames_top_p_top_k() -> None:
    out = cohere_kwargs(GenerationParams(top_p=0.9, top_k=20))
    assert out["p"] == 0.9
    assert out["k"] == 20


def test_extra_passthrough() -> None:
    out = openai_kwargs(GenerationParams(extra={"logit_bias": {"1": 2}}))
    assert out["logit_bias"] == {"1": 2}
