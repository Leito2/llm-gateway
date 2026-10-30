# ADR-0003: One worker by default; resilience state lives in process memory

- **Status:** accepted
- **Date:** 2026-10-29

## Context
The circuit breaker, singleflight and the fallback rate limiter need state. With several uvicorn
workers, each worker is a separate process with its own memory. The options were:
- **Process memory:** simple and fast, but each worker must detect a failing provider on its own.
- **Redis:** shared between workers, but adds a Redis round trip to every request and makes the
  breaker depend on Redis being up — exactly when the gateway is already degraded.

The gateway is async end to end (ADR-6), so one worker serves many concurrent requests. Cloud Run
runs one instance for P3's final test (ADR-0006).

## Decision
- uvicorn runs with **1 worker** by default.
- Breaker state, singleflight and the fallback rate limiter (used when Redis is down) live in
  **process memory**.
- Provider quotas and the budget stay in **Redis**, because they must be shared and must survive
  a gateway restart.

Initial breaker values (tuned after G4 and G10): open after 3 consecutive failures, or > 50% errors
or > 60% slow calls (TTFT > 80% of the alias `ttft_timeout`) over the last 20 calls with at least
10 calls; cooldown 30 s doubling per re-open up to 5 min; 1 trial request when half-open.

## Consequences
- No extra latency or Redis dependency on the hot path of the breaker.
- With N workers (measured in G13), each worker needs about 3 requests to open its own breaker.
  This cost is accepted.
- With 1 worker, the in-memory rate limiter is exact; with N workers it is approximate.
- Moving the breaker to Redis later is a contained change behind the `resilience/` interface.
