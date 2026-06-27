from __future__ import annotations

import pytest

from comparellm.settings import Settings


def test_cors_origins_accepts_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: pydantic-settings must not JSON-decode this env value.
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://a.com,http://b.com")
    assert Settings().cors_allow_origins == ["http://a.com", "http://b.com"]


def test_cors_origins_single_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://only.com")
    assert Settings().cors_allow_origins == ["http://only.com"]


def test_cors_origins_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    assert Settings().cors_allow_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
