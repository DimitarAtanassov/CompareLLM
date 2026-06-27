# CompareLLM - root orchestration Makefile.
# Backend quality gates live in backend/Makefile (make -C backend lint|test|...).

COMPOSE := docker compose

# Local-dev wiring for the backend when run against the dockerized databases.
DB_ENV := \
	MODELS_CONFIG=../config/models.yaml \
	VECTOR_BACKEND=pgvector SESSION_BACKEND=redis \
	DATABASE_URL=postgresql+asyncpg://compare:compare@localhost:5432/compare \
	REDIS_URL=redis://localhost:6379/0

.DEFAULT_GOAL := help

.PHONY: help
help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Full stack (Docker Compose)
# ---------------------------------------------------------------------------
.PHONY: stack
stack: ## Build and run the ENTIRE stack (postgres, redis, ollama, api, ui)
	$(COMPOSE) up --build

.PHONY: stack-detached
stack-detached: ## Run the entire stack in the background
	$(COMPOSE) up --build -d

.PHONY: down
down: ## Stop the stack (keep data volumes)
	$(COMPOSE) down

.PHONY: clean
clean: ## Stop the stack and DELETE data volumes (postgres/redis/ollama)
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail logs from all running services
	$(COMPOSE) logs -f

# ---------------------------------------------------------------------------
# Databases only
# ---------------------------------------------------------------------------
.PHONY: db
db: ## Start the databases (Postgres + Redis) in the background
	$(COMPOSE) up -d postgres redis

.PHONY: db-down
db-down: ## Stop the databases
	$(COMPOSE) stop postgres redis

# ---------------------------------------------------------------------------
# Run individual services locally (on the host)
# ---------------------------------------------------------------------------
.PHONY: backend
backend: ## Run the backend locally with in-memory backends (no DB needed)
	cd backend && MODELS_CONFIG=../config/models.yaml \
		uv run uvicorn comparellm.main:app --reload --port 8080

.PHONY: backend-db
backend-db: ## Run the backend locally wired to the dockerized DBs (needs 'make db')
	cd backend && $(DB_ENV) \
		uv run uvicorn comparellm.main:app --reload --port 8080

.PHONY: frontend
frontend: ## Run the frontend locally (Next.js dev server on :3000)
	cd ui && [ -d node_modules ] || npm install; \
		NEXT_PUBLIC_API_BASE_URL=http://localhost:8080 npm run dev

# ---------------------------------------------------------------------------
# Local dev: databases in Docker, backend + frontend on the host
# ---------------------------------------------------------------------------
.PHONY: dev
dev: db ## Start DBs (Docker) then run backend + frontend together on the host
	$(MAKE) -j2 backend-db frontend
