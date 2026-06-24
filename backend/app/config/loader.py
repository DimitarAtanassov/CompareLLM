"""Load and validate the provider catalogue from ``models.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.config.schema import ModelsConfig
from app.errors import ConfigurationError
from app.log import get_logger

log = get_logger(__name__)


def load_models_config(path: str | Path) -> ModelsConfig:
    """Read, parse, and validate the models configuration.

    Raises:
        ConfigurationError: if the file is missing or malformed.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"models.yaml not found at: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"models.yaml is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("models.yaml must be a mapping at the top level.")

    providers_raw = raw.get("providers") or {}
    if not isinstance(providers_raw, dict):
        raise ConfigurationError("models.yaml 'providers' must be a mapping.")

    # Inject each provider's key into its body so ProviderConfig is self-describing.
    for key, body in providers_raw.items():
        if isinstance(body, dict):
            body.setdefault("key", key)

    try:
        config = ModelsConfig.model_validate({"providers": providers_raw})
    except Exception as exc:  # pydantic ValidationError -> friendly message
        raise ConfigurationError(f"models.yaml failed validation: {exc}") from exc

    log.info(
        "models_config_loaded",
        path=str(config_path),
        providers=sorted(config.providers),
        chat_models=len(config.chat_targets()),
        embedding_models=len(config.embedding_targets()),
    )
    return config
