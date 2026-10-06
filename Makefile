.DEFAULT_GOAL := help
.PHONY: help doctor setup dev up down test lint fmt

help:        ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'
doctor:      ## Check prerequisites (Docker, uv, RAM, ports, Ollama)
	python scripts/doctor.py
setup:       ## Install dependencies with uv
	uv sync --dev
dev:         ## Run the gateway locally with auto-reload
	uv run uvicorn llm_gateway.main:app --reload --port 8080
up:          ## Start gateway + Redis in Docker
	docker compose up -d --build
down:        ## Stop the stack
	docker compose down
test:        ## Run tests
	uv run pytest -q
lint:        ## Lint
	uv run ruff check .
fmt:         ## Format
	uv run ruff format .
