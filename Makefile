.DEFAULT_GOAL := help
.PHONY: help doctor setup dev up up-obs down test eval bench lint fmt

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
up-obs:      ## Start gateway + Redis + Prometheus, Grafana, Jaeger
	docker compose --profile obs up -d --build
down:        ## Stop the stack (all profiles)
	docker compose --profile obs down
test:        ## Run tests
	uv run pytest -q
eval:        ## Nightly checks: G10 availability (the rest of the suite is `make test`)
	uv run python -m bench.availability
bench:       ## Local numbers: G1 overhead, G13 throughput, G9 hit latency, G11 savings, G12 hedging, G14 guardrails
	uv run python -m bench.overhead
	uv run python -m bench.throughput
	uv run python -m bench.hits
	uv run python -m bench.savings
	uv run python -m bench.hedging
	uv run python -m bench.guardrails
lint:        ## Lint
	uv run ruff check .
fmt:         ## Format
	uv run ruff format .
