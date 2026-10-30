# ADR-0004: Budget accounting — integer micro-dollars, three caps, skip paid providers

- **Status:** accepted
- **Date:** 2026-10-29

## Context
ADR-3 in `PLAN.md` sets reserve-then-settle in Redis/Lua; ADR-7 sets fail-closed for paid providers.
Several details were open:
- Floating-point sums in Lua drift, and a Haiku 5.5 request can cost US$0.00005.
- There are three caps (run, per client per day, global per day). Reserving on one and failing on
  another would leave money blocked.
- R05 answered `budget_exceeded` even when a free provider later in the chain could serve.
- A process can die between reserve and settle, and a settle can arrive after the reservation
  expired.

## Decision
- **Units:** all amounts are stored as **integer micro-dollars** (1 USD = 1 000 000). Dollars appear
  only when displayed.
- **One atomic check:** a single Lua script checks the three caps together; if any cap does not fit,
  nothing is reserved. The day starts at 00:00 UTC.
- **Only paid providers reserve.** Free providers never touch the budget; their reference cost
  goes only to metrics. If Redis is down, free providers keep working and paid ones are blocked.
- **Skip instead of fail:** if the reservation does not fit, the gateway skips that paid provider,
  continues with the chain and sets `X-Budget-Skipped`. `402 budget_exceeded` only when no provider
  is left. R05 is rewritten: the gateway SHALL NOT call a paid provider without an accepted
  reservation.
- **Max cost:** input tokens (Anthropic `count_tokens`, fallback `ceil(chars / 3)`) × input price
  (the `cache_write` price when prompt caching is on) + clamped `max_tokens` × output price.
  Adaptive thinking is billed as output inside `max_tokens`, so it is covered.
- **Expiry:** each reservation has a TTL of the alias `total_timeout` + 30 s. A late settle still adds
  the real cost to `spent` and increments `gw_budget_late_settle_total`.
- **Cancelled streams** settle full input plus emitted output; missing `usage` is estimated
  conservatively (rounded up).
- **Redis memory:** `maxmemory-policy volatile-lfu` (only keys with a TTL, i.e. cache entries, are
  evicted; budget keys have none) and AOF `appendfsync everysec`.

## Consequences
- Exact sums: the "0 overspend" test (G3) cannot fail because of rounding.
- Recorded spend is never lower than real spend; in the worst case it is slightly higher.
- Clients get an answer from a free provider instead of an error when the paid budget runs out.
- A test fills the cache up to `maxmemory` and checks that budget keys survive.
