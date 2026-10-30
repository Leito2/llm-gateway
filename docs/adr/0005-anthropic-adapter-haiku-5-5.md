# ADR-0005: Native Anthropic adapter for Claude Haiku 5.5, with prompt caching in core

- **Status:** accepted
- **Date:** 2026-10-29

## Context
`smart_paid` is the only paid alias, used in P3's final test. The model is Claude Haiku 5.5
(`claude-haiku-5-5`). Facts that shape the adapter (verify again before the paid run):
- Price per million tokens for prompts ≤ 100K tokens: US$0.10 input / US$0.50 output; above 100K:
  US$0.50 / US$2.50. Cache read US$0.01 (0.1×), 5-minute cache write US$0.125 (1.25×), 1-hour cache
  write US$0.20 (2×). Minimum cacheable prefix: 512 tokens. 1M context, 128K output.
- Adaptive thinking is on by default (effort `medium`) and is billed as output inside `max_tokens`.
- Non-default `temperature`, `top_p` or `top_k` return 400 (the docs say all models reject them from
  2026-10-21). Prefill returns 400.
- `stop_reason: "refusal"` has no server-side fallback for Haiku.

## Decision
- **Native adapter** (`providers/anthropic.py`), not the OpenAI-compatible one, so prompt caching,
  thinking and refusals are under our control.
- **Prompt caching in core:** the alias enables it (`prompt_cache: true`); the gateway marks the
  system prompt and tools with `cache_control` when they reach the model minimum. Clients opt out
  with `X-Prompt-Cache: off`. The reservation prices input at the `cache_write` rate (ADR-0004).
- **Cheap tier only:** `smart_paid` has `max_input_tokens: 100000`; longer requests get a 400 before
  any call.
- **Sampling parameters:** `temperature`, `top_p` and `top_k` are stripped, `X-Params-Dropped` lists
  them, and the request is not an error.
- **Effort per alias:** `effort` in `routes.yaml` (`smart_paid: low`). A client `reasoning_effort` is
  honoured but clamped to at most the alias value. Thinking text is never forwarded; thinking events
  count as liveness for the stall timeout.
- **Refusal** maps to `finish_reason: "content_filter"`: no fallback, not counted by the breaker,
  not cached.
- **Testing at $0:** fixtures written from the documented event stream now; a few real streams are
  recorded once with `GW_RUN_BUDGET_USD=0.10` before the final test. Whether `count_tokens` is free
  is verified in M7.

## Consequences
- The adapter is larger than an OpenAI-compatible config entry, but prompt caching is measurable
  (share of input tokens served from the provider cache).
- Client code written for OpenAI keeps working: unsupported parameters are dropped, not rejected.
- Prices and parameter rules change; `prices.yaml` and this ADR are re-checked before the paid run.
