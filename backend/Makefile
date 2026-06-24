sources = app tests

.PHONY: prepare ## Sync the locked virtual environment
prepare:
	uv sync

.PHONY: lintable ## Apply auto-formatting and auto-linting
lintable: prepare
	uv run ruff format $(sources)
	uv run ruff check --fix $(sources)

.PHONY: lint ## Run linting checks
lint: prepare
	uv lock --check
	uv run ruff format --check $(sources)
	uv run ruff check $(sources)
	uv run mypy app

.PHONY: test ## Run tests and coverage reports
test: prepare
	uv run coverage run -m pytest -m "not contract and not db"
	uv run coverage report

.PHONY: test-contract ## Run contract tests
test-contract: prepare
	ENVIRONMENT=contract uv run -m pytest -m "contract"

.PHONY: test-integration ## Run integration tests
test-integration: prepare
	uv run -m pytest -m "integration"

.PHONY: test-db ## Validate against the internal Postgres image (needs Docker)
test-db: prepare
	uv run -m pytest -m "db" -v
