# 🛡️ llm-gateway

> An OpenAI-compatible LLM gateway written from scratch in Python — multi-provider fallback (Groq, Gemma 4 via
> Google AI Studio, Ollama, Anthropic), mandatory circuit breakers, hedged requests, **atomic budget enforcement**,
> a layered semantic cache with calibrated thresholds, edge guardrails, and SSE streaming with end-to-end cancellation.
> Successor of my Go [llm-edge-gateway](https://github.com/Leito2/llm-edge-gateway) — every capability kept, its limits fixed.
> Shared by four portfolio systems: real-time fraud detection, a System 1/System 2 router, a live RAG platform,
> and a GraphRAG multi-agent research system.

**Status:** 🟡 M0 bootstrap (structure, configs, CI, health endpoint). See [`PLAN.md`](PLAN.md) for the full design (Spanish).

## TL;DR — Results at a Glance
_To be filled with measured numbers (overhead p95, budget under concurrency, cache hit rate) — milestone M9._

## Part I — The Big Picture
### 1. The Problem: every service calling LLMs on its own
### 2. Core Concepts Primer
- API gateways and the OpenAI-compatible contract
- Timeouts per phase, retries with jitter, circuit breakers, bulkheads
- Budgets under concurrency: reserve → settle
- Exact vs semantic caching (and false hits): namespaces, calibrated thresholds, lexical guards, verifiers
- Hedged requests, adaptive concurrency and load shedding
- Edge guardrails: prompt injection and PII
- Server-Sent Events and cancellation
### Key technologies at a glance
- **OpenAI-compatible API** — any client or SDK works without changes.
- **Circuit breakers and fallback** — if a provider fails, traffic moves to the next one automatically.
- **Semantic cache** — similar questions are answered from cache, saving cost and time.
- **Atomic budget in Redis** — spending can never exceed the limit, even under heavy concurrency.
- **SSE (Server-Sent Events)** — answers are streamed token by token, and stop if the client disconnects.
### 3. Architecture · 4. Design Decisions · 5. Journey of a Request

## Part II — Components (Concept → How it works here → Technical details)
## Part III — Resilience and Cost Control in Depth
## Part IV — Proof (G1–G15, EARS traceability, Go predecessor and LiteLLM comparison)
## Part V — Run It Yourself

### Prerequisites
Docker Desktop (WSL2), [uv](https://docs.astral.sh/uv/), Python 3.12 (installed by uv), optional Ollama for local models.

```bash
python scripts/doctor.py      # or: make doctor
uv sync --dev                 # or: make setup
uv run pytest -q              # M0 tests: health, schemas, routing config
make up                       # gateway + Redis in Docker
curl localhost:8080/v1/models
```

## Part VI — Reflection (lessons, limitations, why not LiteLLM)

## License
MIT
