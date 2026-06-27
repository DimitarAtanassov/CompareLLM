# CompareLLM Backend

FastAPI service that streams multi-model chat comparisons and runs embedding-based
semantic search across many providers using native provider SDKs (no LangChain or
LangGraph).

See the [project README](../ReadMe.md) for the full architecture, request/response
contracts, and run instructions.

This project is managed with [uv](https://docs.astral.sh/uv/).

## Local development

```bash
cd backend
uv sync                       # create the locked .venv and install all deps

export MODELS_CONFIG=../config/models.yaml
uv run uvicorn comparellm.main:app --reload --port 8080
```

## Quality gates (Makefile)

```bash
make lintable   # auto-format + auto-fix (ruff format, ruff check --fix)
make lint       # uv lock --check, ruff format --check, ruff check, mypy
make test       # pytest with coverage (excludes contract/db markers)

make test-contract     # marker: contract
make test-integration  # marker: integration
make test-db           # marker: db (needs Docker/Postgres)
```

Linting uses a strict ruff ruleset (pyflakes, pycodestyle, mccabe, isort,
pep8-naming, pyupgrade, flake8-annotations/bandit/bugbear/comprehensions/
pytest-style, pylint, perflint, and more) configured in `pyproject.toml`.
