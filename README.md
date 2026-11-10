# 🛡️ llm-gateway

> An OpenAI-compatible LLM gateway written from scratch in Python — multi-provider fallback (Groq, Gemma via
> Google AI Studio, Ollama, Anthropic Haiku 5.5), circuit breakers on every provider, **atomic budget enforcement**,
> a layered semantic cache with lexical guards, structured-output repair, SSE streaming with end-to-end
> cancellation, and Prometheus/OpenTelemetry observability.
> Successor of my Go [llm-edge-gateway](https://github.com/Leito2/llm-edge-gateway) — every capability kept, its limits fixed.

**Status:** `v1.1` — milestones M0–M9 done, 366 tests. M4 (semantic cache) is complete except the items listed under
[Limitations](#part-vi--reflection): calibrated thresholds and a gray-zone verifier model.
[`PLAN.md`](PLAN.md) holds the full design (Spanish); [`GLOSSARY.md`](GLOSSARY.md) fixes the vocabulary.

## TL;DR — Results at a Glance

Measured on a Windows 11 laptop (8 cores), gateway + upstream + load generator on the same machine. Raw files are in
[`bench/results/`](bench/results/).

| What | Result | Gate |
|---|---|---|
| Tests | **366 passing**, lint clean, every EARS requirement R01–R23 traced to a test | CI on every push |
| Budget under concurrency | 50 concurrent requests against a cap that fits 10 → **exactly 10 reserved, 0 overspend** (memory store and Redis Lua) | G3 ✅ |
| Availability with a sick primary | 10,000 requests, primary failing 40% of calls (503 / 429 / slow) → **99.98% success** (9,998 × 200) | G10 ✅ (≥ 99.9%) |
| Overhead, no cache | **+6.9 ms p50 / +10.4 ms p95 / +14.0 ms p99** over calling the upstream directly at 100 req/s | G1 ⚠️ missed the < 5 ms p95 target on this setup |
| Overhead, semantic cache *miss* (e5-small embedding + KNN) | **+17.5 ms p50 / +27.7 ms p95** on top of the no-cache path, at 20 req/s | G1 (second number) |
| Exact-cache throughput | **≈ 1,200 req/s** on one worker | G13 (the Go figure is not reproduced; see below) |
| Provider calls saved by the cache | exact hits cost 0 provider calls, 0 quota, 0 bulkhead slots; synthetic Zipf traffic: **70–94 %** of reference cost saved | G11 🟡 synthetic traffic only; real traffic not measured |

What these numbers do **not** say: the load generator is Python/httpx on the same box, so absolute latencies include
its own overhead — read the *difference* between "direct" and "through the gateway". Windows has no `SO_REUSEPORT`, so
extra uvicorn workers did not add throughput here (1 / 2 / 4 workers: 1,232 / 1,182 / 1,246 req/s); that
comparison needs Linux. The < 5 ms p95 target was **not met** in this environment, and I publish it as it came out.

## Part I — The Big Picture

### 1. The problem: every service calling LLMs on its own
Four portfolio systems (fraud detection, a System 1/System 2 router, a live RAG platform, a GraphRAG research agent)
all need LLMs. Without a shared layer each one re-implements provider fallback, retries, rate-limit handling, cost
control and caching — badly, and differently. The gateway is that layer: one OpenAI-compatible endpoint, many providers
behind it, with the failure handling written once and tested hard.

### 2. Core concepts in one minute
- **Alias → fallback chain.** Clients ask for an *alias* (`fast`, `smart`, `judge`, `vision`, `smart_paid`…), never a
  model. `config/routes.yaml` maps each alias to an ordered chain of `{provider, model}` links plus a policy
  (timeouts, cache, priority, token caps). Every chain ends in a local model or the `mock`, so the gateway still answers offline.
- **Timeouts per phase, retries with jitter, circuit breakers, bulkheads.** A request has a time-to-first-token timeout, a
  stall timeout and a total timeout. Connection errors and 5xx get one jittered retry. A breaker per provider opens after
  3 consecutive failures, >50% errors, or >60% slow calls. A bulkhead caps concurrent calls per provider.
- **Retry only before the first token.** Once a token reached the client the gateway never retries or falls back
  (ADR-4); the stream ends with a typed `error` event and no `[DONE]`.
- **Budget = reserve, then settle.** Before a paid call the gateway atomically *reserves* the worst-case cost (in integer
  micro-dollars) against three caps (run, client-day, global-day), then *settles* the real cost. If it does not fit, that
  link is skipped and the chain continues; `402 budget_exceeded` only when nothing is left.
- **Exact vs semantic cache, and false hits.** An exact hit needs identical input. A semantic hit needs a similar
  question — risky, because "refund of $50" and "refund of $500" embed almost identically. So semantic hits must pass
  calibrated thresholds **and** lexical guards (numbers, currency, dates, codes, negation).
- **SSE and cancellation.** Streams are Server-Sent Events. If the client leaves, the upstream call is cancelled and what
  was consumed is settled.

### 3. Architecture

```
client ──► auth ─► rate limit ─► admission (shed) ─► size guardrail ─► cache (exact → semantic)
                                                                          │ miss
                                       singleflight ◄─────────────────────┘
                                            │
                        executor:  per link  breaker → quota → bulkhead → budget reserve
                                            │          timeouts per phase · 1 retry · ≤ 5 calls
                                            ▼
                         providers: Ollama · Groq · Google AI Studio · Anthropic (native) · mock
                                            │
              structured-output check/repair ◄┘   → settle budget → cache write-back → metrics/log/trace
```

State lives in process memory (breakers, singleflight, in-memory cache) with **one uvicorn worker**
([ADR-0003](docs/adr/0003-single-worker-in-process-state.md)); the budget and rate limit can use Redis.

### 4. Journey of a request
1. `Authorization: Bearer` → client (SHA-256 compared in constant time); rpm check; alias must be in `allowed_aliases`.
2. Admission control: when busy, `batch` requests are shed before `interactive` ones (`X-Priority` can only lower).
3. Size/shape guardrail (messages, images, body bytes, `max_input_tokens`).
4. Cache: exact key, then semantic KNN + thresholds + guards. A hit costs no provider call.
5. Singleflight: identical cacheable requests in flight share one provider call.
6. Executor walks the chain; the first link that answers wins. Headers say what happened.
7. Structured output (if requested) is validated, repaired once, and re-emitted as SSE when streaming.
8. Budget settles, the answer is cached, one JSON access-log line is written, metrics and the trace span are closed.

## Part II — Components

| Package | What it does |
|---|---|
| `api/` | FastAPI routes: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/healthz`, `/readyz`, `/stats`, `/metrics`, `/admin/*`, chat UI at `/`; OpenAI-format errors; opt-in CORS |
| `routing/` | `routes.yaml` models and chain resolution |
| `providers/` | `Provider` protocol, `OpenAICompatProvider` (Ollama, Groq, Google), native `AnthropicProvider` (Messages API), `MockProvider` |
| `resilience/` | executor, circuit breaker, bulkhead, admission, singleflight, retry, lifecycle (SIGTERM drain) |
| `quotas/` | outbound token buckets per provider (rpm/tpm/rpd) so the gateway switches before a 429 |
| `budget/`, `pricing/` | micro-dollar prices, atomic reserve/settle (memory and Redis Lua), per-scope caps |
| `cache/` | keys + namespaces, lexical guards, embedder protocol (hash, ONNX e5-small), in-memory backend, calibration harness |
| `structured/` | JSON / JSON-Schema validation and repair |
| `security/` | client keys, admin key, per-client rate limit |
| `guardrails/` | size and shape limits; regex injection / PII heuristics; optional Prompt Guard 2 ONNX classifier (needs model files) |
| `telemetry/` | Prometheus metrics, JSON access log (Presidio-redacted prompt when `GW_LOG_PROMPTS=true`), OpenTelemetry tracing (Jaeger and/or Langfuse over OTLP) |

## Part III — Resilience and Cost Control in Depth

**Failure handling (M3, M6).** Gates per link, in order: breaker → outbound quota → bulkhead → budget. Gateway timeouts
count as `transient` with status 408 and are never retried. A stream that stalls after its first token ends with a typed
`stream_stalled` event. SIGTERM flips `/readyz` to 503, drains streams for up to 8 s, then cuts the rest with a typed
`server_shutting_down` event. Keep-alive `: ping` comments go out every 10 s of silence; thinking events count as
liveness for the stall timer but are never forwarded.

**Budget (M5, [ADR-0004](docs/adr/0004-budget-accounting.md)).** Integer micro-dollars (`Decimal`, rounded up), three
caps checked together in one Lua script, reservations that expire logically (a deadline in a sorted set, no key TTLs so
`volatile-lfu` can never evict budget keys), late settles still add to spent. Free providers never reserve. If the store is
down, paid providers fail closed and free ones continue.

**Semantic cache (M4, [ADR-0002](docs/adr/0002-semantic-cache-own-implementation-not-langcache.md)).** Namespace =
client × alias × system-prompt hash × `X-Prompt-Version` × `response_format` × embedder. Thresholds live per embedder in
`config/cache.yaml`; **while an embedder is uncalibrated the cache only observes** (counts shadow hits, serves nothing).
Degraded, truncated, refused or empty answers are never stored; `X-Cache-Scope: private` bypasses both layers.

**Anthropic adapter (M7, [ADR-0005](docs/adr/0005-anthropic-adapter-haiku-5-5.md)).** `temperature`/`top_p` are dropped
(`X-Params-Dropped`), `reasoning_effort` is clamped to the alias, the system prompt gets `cache_control` when it reaches
512 tokens, usage is normalized to the OpenAI convention, and a `refusal` becomes `finish_reason: "content_filter"`. It is
registered only when `GW_ANTHROPIC_API_KEY` is set **and** `GW_RUN_BUDGET_USD > 0`.

## Part IV — Proof

### Test gates (PLAN §10)

| Gate | Status | Evidence |
|---|---|---|
| G1 overhead | ⚠️ measured, target missed | `bench/overhead.py` → +10.4 ms p95 no cache; +27.7 ms p95 more on a semantic miss |
| G3 budget under concurrency | ✅ | `tests/test_budget.py` (memory + Redis Lua), `test_requirements.py::test_req_06` |
| G4 fallback | ✅ | `tests/test_resilience.py::test_g4_*` |
| G5 cancellation + Go regression | ✅ | `tests/test_streaming.py` (real uvicorn) |
| G6 semantic cache quality | 🟡 harness + 42 seed pairs; ~1,000 labeled pairs pending | `scripts/calibrate_cache.py` |
| G7 contracts | ✅ | `tests/test_openai_sdk.py` (official OpenAI SDK) |
| G8 chaos | 🟡 mock-based; Toxiproxy/Redis chaos pending | `tests/test_resilience.py` |
| G9 hit latency | ✅ exact p95 2.0 ms; semantic p95 11.5 ms (target <60 ms), e5-small, threshold set for the run only | `bench/hits.py` |
| G10 availability | ✅ 99.98% | `bench/availability.py`, nightly workflow |
| G11 cost saved | 🟡 synthetic Zipf, exact cache only: 70–94 % of reference cost saved depending on traffic shape; real P2/P3 traffic not available | `bench/savings.py` |
| G13 throughput | ✅ measured ≈ 1,200 req/s | `bench/throughput.py` |
| G15 quotas | ✅ | `tests/test_quotas_and_fallback.py` |
| G12 hedging | ✅ p99 1,015 ms → 131 ms (7.7×) for +6.5% extra calls | `bench/hedging.py` (local only) |
| G14 guardrails | 🟡 hand-written set (94 rows): injection recall 100 % direct / 0 % paraphrase-obfuscation, 4.7 % false positives; PII recall 100 %, precision 93 % | `bench/guardrails.py` |

### EARS traceability (PLAN §11)

| ID | Requirement (short) | Test |
|---|---|---|
| R01 | OpenAI-format answer + `X-Gateway-Provider`, `X-Alias`, `X-Request-Id` | `test_chat.py::test_req_01_response_shape_and_headers` |
| R02 | Missing/wrong key → 401, no provider called | `test_auth.py::test_req_02_bad_or_missing_key_is_401_and_calls_no_provider` |
| R03 | Primary fails before first token → next link | `test_quotas_and_fallback.py::test_req_03_transient_failure_falls_back_and_reports_the_depth` |
| R04 | Open breaker → provider skipped | `test_resilience.py::test_r04_an_open_breaker_is_skipped_without_calling_the_provider` |
| R05 | No paid call without a reservation; `budget_exceeded` when nothing is left | `test_requirements.py::test_req_05_*` |
| R06 | Settled spend never exceeds the cap | `test_requirements.py::test_req_06_*` |
| R07 | Semantic/exact hit with `X-Cache-Layer` | `test_cache.py::test_r07_*` |
| R08 | Number/date/negation difference → miss | `test_cache.py::test_r08_*` |
| R09 | Tag invalidation | `test_cache.py::test_r09_*` |
| R10 | Client disconnect cancels upstream | `test_streaming.py::test_client_disconnect_closes_the_upstream_and_frees_the_slots` |
| R11 | No retry after first token; typed `error` event | `test_resilience.py::test_a_stream_that_stalls_after_its_first_token_ends_with_a_typed_error_and_no_fallback` |
| R12 | Store down → paid rejected, free keep serving | `test_requirements.py::test_req_12_*` |
| R13 | Exhausted quota → next provider, no 429 | `test_quotas_and_fallback.py::test_req_13_g15_*` |
| R14 | Batch shed first | `test_resilience.py::test_r14_*` |
| R15 | Invalid structured output → `invalid_structured_output` | `test_structured.py::test_every_provider_invalid_is_a_502_within_the_attempt_limit` |
| R16 | No prompts in logs | `test_observability.py::test_one_json_line_per_request_and_never_the_prompt` |
| R17 | Unknown alias → 404 with valid aliases | `test_chat.py::test_req_17_*` |
| R18 | Alias not allowed → 403 | `test_auth.py::test_req_18_*` |
| R19 | `max_tokens` clamped + `X-Max-Tokens-Clamped` | `test_chat.py::test_req_19_*` |
| R20 | Refusal → `content_filter`, no fallback, not cached | `test_requirements.py::test_req_20_*` |
| R21 | `X-Cache-Scope: private` bypasses the cache | `test_cache.py::test_r21_*` |
| R22 | `LOG_PROMPTS=true` without Presidio → refuses to start | `test_observability.py::test_log_prompts_without_presidio_refuses_to_start` |
| R23 | SIGTERM drain | `test_resilience.py::test_r23_*` (three tests) |

### Headers

| Response header | Meaning |
|---|---|
| `X-Request-Id` | Trace id (the client's `X-Request-ID` if safe, otherwise new) |
| `X-Alias` | Resolved alias |
| `X-Gateway-Provider` | `provider:model` that actually answered |
| `X-Gateway-Fallback-Depth` | Links tried (1 = first link answered; 0 = cache hit) |
| `X-Gateway-Singleflight` | `shared` when the answer came from another identical in-flight request |
| `X-Cache-Layer` | `exact` or `semantic` on a cache hit |
| `X-Cache-Similarity` | Cosine similarity of a semantic hit |
| `X-Max-Tokens-Clamped` | `max_tokens` was cut to the alias cap (the cap) |
| `X-Params-Dropped` | Parameters the provider rejects and the gateway removed (`temperature`…) |
| `X-Effort` | Effective reasoning effort after clamping |
| `X-Structured-Repaired` | JSON was cleaned (code fence / prose removed) without a new call |
| `X-Budget-Skipped` | Paid provider skipped because the budget did not fit |
| `Retry-After` | On 429 / 503 |

Request headers: `Authorization: Bearer`, `X-Priority: batch`, `X-Cache-Scope: private`, `X-Cache-Tags`,
`X-Prompt-Cache: off`, `X-Prompt-Version`, `X-Request-ID`, `traceparent`.

### Versus the Go predecessor
The Go gateway already had fallback, a budget, an exact/semantic cache and SSE. What this version changes: the budget is
**atomic under concurrency** (the Go one could overspend), breakers are mandatory on every provider with three triggers,
retries stop at the first token, the semantic cache has guards and an *observation mode* instead of a hand-picked
threshold, and the stream-survives-the-handler bug is covered by a real-server regression test. What it does **not** beat
is raw speed: Go's 10K+ req/s on the cache path is not reachable from Python (≈ 1.2K here).

## Part V — Run It Yourself

### Prerequisites
[uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it). Docker only for the Redis / observability stack;
Ollama optional for local models. Everything below runs offline against the built-in `mock` provider.

```bash
uv sync --dev                                   # install
uv run pytest -q                                # 366 tests
python scripts/new_key.py me fast smart         # prints a key ONCE + the clients.yaml entry to paste
```

Create `config/clients.yaml` (git-ignored) with the entry the script printed, then:

```bash
uv run uvicorn llm_gateway.main:app --port 8080
```

```bash
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"hello"}]}' -i
```

Open <http://localhost:8080/> for the chat UI (paste your key; response headers show as pills). <http://localhost:8080/guide> serves the project guide (phases, features, run instructions) with a real chat that calls this gateway. The official SDK works
unchanged: `OpenAI(base_url="http://localhost:8080/v1", api_key=KEY)`.

| Setting (env, prefix `GW_`) | Purpose |
|---|---|
| `ENV` | `dev` (default), `test`, `prod`; the `mock` provider exists only in dev/test |
| `GROQ_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_BASE_URL` | Enable those providers (a provider with no key is simply not registered) |
| `ANTHROPIC_API_KEY` + `RUN_BUDGET_USD` | Both needed to enable paid Anthropic; `RUN_BUDGET_USD=0` disables paid providers |
| `GLOBAL_DAY_BUDGET_USD`, `BUDGET_BACKEND` | Global day cap; `memory` (default) or `redis` |
| `CACHE_BACKEND`, `EMBEDDER` | `memory`/`redis`/`off`; `e5-small` (needs `uv sync --extra semantic` + model files), `hash`, `off` |
| `ADMIN_KEY_SHA256` | Hash of the admin key (`python scripts/new_key.py --admin`) guarding `/stats`, `/metrics`, `/admin/*` |
| `CORS_ORIGINS` | Comma-separated origins (or `*`) allowed to call from a browser app on another origin |
| `OTEL_ENDPOINT` | OTLP/HTTP traces (`uv sync --extra otel`) |
| `STREAM_PING_S`, `MAX_IN_FLIGHT`, `RATE_LIMIT_BACKEND`, `MAX_BODY_BYTES` | Tuning |

```bash
make up            # gateway + Redis (Docker)
make up-obs        # + Prometheus, Grafana (dashboard provisioned), Jaeger
make bench         # G1 overhead and G13 throughput (local, slow)
make eval          # G10 availability (also runs nightly in CI)
```

## Part VI — Reflection

**What worked.** Writing the failure model first (error kinds, "retry only before the first token", reserve-then-settle)
made the rest mechanical; every EARS requirement has a test, and the real-uvicorn tests caught things TestClient hides.

**Limitations (honest list).**
- Semantic thresholds are **uncalibrated**: the cache runs in observation mode until ~1,000 labeled pairs exist and the
  precision/coverage curves are measured for several embedders. The gray-zone verifier model is not built. The Redis cache backend (`GW_CACHE_BACKEND=redis`) passes its tests on fakeredis and, with the Redis client swapped for a real Redis 8 server, the same 52 cache and budget tests (including the Lua budget scripts); it ranks vectors in Python instead of using the Redis 8 Query Engine (HNSW). The `make up-obs` stack (gateway, Redis, Prometheus, Grafana, Jaeger) was brought up in Docker: Prometheus scrapes `/metrics` with the admin key, Jaeger receives the traces, and the per-client rate limit lives in Redis. A cache hit through that stack was not exercised, because no real provider (Ollama or a key) was available and the mock is `degraded`, which is never cached.
- G11 uses synthetic Zipf traffic, so the PLAN's 30–40 % claim is untested on real traffic; G9's semantic threshold was set
  just for the run (G6 is open). The Toxiproxy chaos suite is not measured yet.
- The Anthropic adapter is tested against fixtures written from the documented API, not against real streams.
- The < 5 ms p95 overhead target was missed on the Windows test machine.
- Presidio runs for real (`uv sync --extra pii`) and redacts the prompt in the access log when `GW_LOG_PROMPTS=true`, but with a
  blank spaCy pipeline: only its pattern recognizers work (email, card, phone, IBAN, SSN, IP, URL, plus a Spanish DNI one), so
  names and places are logged as written. It does not redact outbound traffic.
- Real providers: Google AI Studio (`gemma-4-31b-it`, free tier) was called through the gateway: a JSON answer came back, the
  repeat was an exact cache hit, and a real upstream `HTTP 500` on streaming opened the breaker and fell down the chain. The
  provider's SSE parsing was also run against `gemini-2.5-flash`. Gemma 4 returns its reasoning as the message text
  (`<thought>…`) and reports `completion_tokens: 0`; the gateway passes that through. Groq was called too (`openai/gpt-oss-20b` for `fast`, JSON and SSE through the gateway). Its old model ids
  (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) answered 404, so `config/routes.yaml` now uses `openai/gpt-oss-20b` and
  `openai/gpt-oss-120b`; those models return a `reasoning` field next to `content`. Anthropic was not called (no key). xAI (Grok,
  a different service from Groq with `xai-` keys) has no credits and no adapter yet.
- Prompt Guard 2 is a gated Meta model; the ONNX classifier code (`guardrails/promptguard.py`) is tested only with a fake session
  and has never run on real weights. Without `models/prompt-guard/` the regex heuristic is all there is (default `monitor`). On a small hand-written set it catches every direct phrase and none of the paraphrased
  or obfuscated ones, and flags 3 of 40 benign prompts ("developer mode" on Android, "do anything now" in a song).
- Langfuse export (`GW_LANGFUSE_*`) goes through its OTLP/HTTP endpoint (its legacy ingestion API is deprecated); the unit tests only
  check the endpoint, Basic auth and `x-langfuse-ingestion-version` header it is configured with. It was also run against a self-hosted
  Langfuse 4.75 (official compose file): `gateway.request` and `provider.attempt` observations arrived with the model and the token usage
  (input/output). Langfuse Cloud and prompt/cost views were not checked. Hot reload covers routes, client keys and prices, not providers or budget caps.
- Hedging, EWMA reordering and the adaptive bulkhead are opt-in per alias / provider and tested in-process only.

**Why not LiteLLM.** LiteLLM is the reference baseline (G2, optional). This project exists to own the failure semantics
(atomic budget, guarded semantic cache, no retry after the first token) and to understand them end to end.

## License
MIT
