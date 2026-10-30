# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

OpenAI-compatible LLM gateway in Python (FastAPI), successor of the Go `llm-edge-gateway`. **Only milestone M0 is done**: `GET /healthz`, `GET /v1/models`, Pydantic schemas, routing-config loading, and CI. `/v1/chat/completions`, providers, resilience, cache, budget, guardrails and telemetry are empty packages (`__init__.py` only) awaiting later milestones.

[`PLAN.md`](PLAN.md) (Spanish, ~600 lines) is the source of truth for design: §1.6 core vs stretch scope, §2 architecture and ADR-1…15, §3.1 provider error kinds, §3.2 client contract, §4 resilience, §5 budget, §6 semantic cache, §8.5 headers, §11 EARS requirements R01–R23 (each must get a test), §13 target repo layout, §14 milestones M0–M9. Check the relevant section and the milestone table before implementing a component; the package directories under `src/llm_gateway/` already mirror the §13 layout. README is in English and written progressively (concepts first, technical detail last); PLAN.md is Spanish. File ADRs live in `docs/adr/` (0002–0006 hold the detailed decisions on cache, single-worker state, budget accounting, the Haiku 5.5 adapter and the P3 deployment). [`GLOSSARY.md`](GLOSSARY.md) fixes one term per concept — use those names in code and docs. Only build stretch items (§1.6) after core is done.

## Commands

Uses `uv` and Python 3.12. `make` targets wrap these (on Windows without `make`, run the underlying commands).

```bash
uv sync --dev                                   # install (make setup)
uv run pytest -q                                # all tests (make test)
uv run pytest tests/test_schemas.py::test_name  # single test
uv run ruff check .                             # lint (make lint); CI runs lint then tests
uv run ruff format .                            # format (make fmt)
uv run uvicorn llm_gateway.main:app --reload --port 8080   # dev server (make dev)
make up / make down                             # gateway + Redis via docker compose
make up-obs                                     # + Prometheus, Grafana, Jaeger (compose profile obs)
python scripts/doctor.py                        # check Docker, uv, RAM, ports, Ollama
```

Ruff: line length 110, rules `E,F,I,UP,B,ASYNC`. `pytest-asyncio` runs in `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed). `mypy` is a dev dependency but not wired into CI.

## Architecture notes (what spans several files)

- **Alias → chain routing.** Clients request an *alias* (`fast`, `smart`, `judge`, `vision`, `extract`, …), never a model. `config/routes.yaml` maps each alias to an ordered fallback chain of `{provider, model}` plus per-alias policy (timeouts per phase, `cache`, `semantic_cache`, `cache_ttl_s`, `hedge`, `priority`, `allow_degraded`, and for `smart_paid` `max_input_tokens`, `effort`, `prompt_cache`; chain links may carry `degraded: true`); `routing/config.py` validates it (`RoutesConfig`/`AliasPolicy`) — several of these keys are not modelled yet and are silently ignored until their milestone. `provider` keys refer to `config/providers.yaml`. Every chain ends in a local model (`ollama`) and/or the `mock` provider so the gateway answers offline. `/v1/models` lists aliases.
- **Provider interface.** `providers/base.py` defines the `Provider` Protocol (`complete` → OpenAI `chat.completion` dict with `usage`; `stream` → `chat.completion.chunk` dicts, last one carrying `usage`) and `ProviderError(retryable=…)`, which the resilience layer uses to decide retry vs. fallback (PLAN §3.1 replaces `retryable` with a 3-value `kind`: transient / provider / client). Planned adapters: `mock` (only when `GW_ENV` is dev/test), one generic `OpenAICompatProvider` (Ollama via `/v1`, Groq, Google AI Studio), and a native `anthropic` adapter for `claude-haiku-5-5` (ADR-10, ADR-0005).
- **Config surface.** `settings.py` (`pydantic-settings`, env prefix `GW_`, reads `.env`; see `.env.example`) holds file paths and secrets. `create_app(settings)` in `main.py` is the app factory (tests call it directly with `TestClient`); a module-level `app = create_app()` is also created at import, so importing `llm_gateway.main` loads `config/routes.yaml` relative to the CWD.
- **Design invariants from PLAN.md to preserve:** retry/hedge only *before* the first token is emitted (ADR-4); budget is reserve-max-then-settle atomically in Redis/Lua, in integer micro-dollars, only for paid providers, skipping to the next provider when it does not fit, and fails closed for paid providers when Redis is down (ADR-3, ADR-7, ADR-0004); uvicorn runs 1 worker and breaker/singleflight state is in process memory (ADR-0003); semantic cache is on by default for eligible aliases, opt-out per alias (ADR-8); every provider call goes through a circuit breaker (ADR-9); async end-to-end with cancellation propagated to upstream, ONNX inference in a bounded thread pool (ADR-6).
- **Data policy:** free-tier providers (Groq, Google AI Studio) receive only synthetic/public data or user-consented data; `anthropic` is only enabled for the final P3 test (`GW_RUN_BUDGET_USD=0` disables paid providers). Dev spend target is $0.
