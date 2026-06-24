from __future__ import annotations

from pathlib import Path

import pytest
from app.config.loader import load_models_config
from app.errors import ConfigurationError

VALID = """
providers:
  openai:
    type: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    models:
      - gpt-4o
    embedding_models:
      - text-embedding-3-small
"""

INVALID_TYPE = """
providers:
  weird:
    type: not-a-real-provider
    models: [foo]
"""


def test_load_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(VALID)
    config = load_models_config(path)

    assert "openai" in config.providers
    assert config.providers["openai"].key == "openai"
    assert config.chat_targets() == ["openai:gpt-4o"]
    assert config.embedding_targets() == ["openai:text-embedding-3-small"]


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError):
        load_models_config("/does/not/exist.yaml")


def test_invalid_provider_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(INVALID_TYPE)
    with pytest.raises(ConfigurationError):
        load_models_config(path)
