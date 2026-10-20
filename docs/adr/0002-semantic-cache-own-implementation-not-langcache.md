# ADR-0002: Build the semantic cache on local Redis 8, not on Redis LangCache

- **Status:** accepted
- **Date:** 2026-10-19

## Context
Redis LangCache is a managed semantic-cache service (REST API + Python SDK): it embeds the prompt,
runs the vector search and returns the cached response. It supports `attributes` filters,
a per-request `similarityThreshold`, `exact` + `semantic` search strategies, and deletion by
entry ID or by attributes. As of October 2026 it is in preview: public preview on Redis Cloud
(final GA pricing TBD), and self-managed only on Kubernetes as a private preview with a license key.

Evaluated against the constraints in `PLAN.md`:
- **$0 dev spend and local-first run** (§12): no `docker compose` deployment; the cloud preview may change price or limits.
- **Offline operation** (§0, `config/routes.yaml`): every chain ends in a local model; a cloud cache adds a network dependency on every request.
- **Hit latency** (G9, p95 < 60 ms): each lookup adds an HTTPS round trip plus a remote embedding.
- **False-hit defenses** (§6.3): the embedder is chosen by the service, so the two-embedder comparison and per-embedder threshold calibration are not possible. Lexical guards and the gray-zone verifier would have to run as a client-side post-filter. Deletion by attributes requires all attributes to match, which does not fit multi-tag invalidation (`X-Cache-Tags: doc:1,doc:7`, §6.4).
- **Data policy** (ADR-11): prompts, including user-consented P4 vault excerpts, would go to one more third party.
- **Portfolio goal** (§1.5): the semantic cache is the main differentiator measured by G6, G9 and G11; delegating it removes that evidence.

## Decision
The semantic cache is implemented in the gateway on the open-source Redis 8 Query Engine, as specified
in §6. Its storage/search goes behind a small `SemanticCacheBackend` interface designed in M4, with
`redis_local` as the only implementation and default.

LangCache is kept as an **optional baseline** for M9, at the same level as LiteLLM and the Go predecessor:
if it is still free and accessible then, run the same labeled pairs (G6) and Zipf replay (G9, G11)
against it and publish the comparison. It is not added as a dependency before that.

## Consequences
- Full control over embedder, thresholds, guards, verifier and tag invalidation; G6/G9/G11 measure our own design.
- We operate the index ourselves (memory limits, HNSW parameters, LFU eviction).
- Adding a `langcache` backend later is a contained change behind the M4 interface, not a redesign.
