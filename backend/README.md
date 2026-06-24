# CompareLLM Backend

FastAPI service that streams multi-model chat comparisons and runs embedding-based
semantic search across many providers using native provider SDKs (no LangChain or
LangGraph).

See the [project README](../ReadMe.md) for the full architecture, request/response
contracts, and run instructions.

## Local development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export MODELS_CONFIG=../config/models.yaml
uvicorn app.main:app --reload --port 8080
```

## Quality gates

```bash
ruff check app tests
mypy app
pytest
```
