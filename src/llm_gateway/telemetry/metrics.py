"""Prometheus metrics (PLAN §8.1). One registry per app, so tests can build many apps.

Labels are bounded on purpose: aliases and providers come from config, clients from clients.yaml,
and nothing here ever carries prompt text or user ids."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from llm_gateway.resilience.breaker import State

_TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)
_STATE_VALUE = {State.CLOSED: 0, State.HALF_OPEN: 1, State.OPEN: 2}


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "gw_requests_total", "Requests by outcome", ["alias", "provider", "status"], registry=r
        )
        self.ttft = Histogram(
            "gw_ttft_seconds",
            "Time until the provider's first token (whole call when not streaming)",
            ["alias", "provider"],
            buckets=_TTFT_BUCKETS,
            registry=r,
        )
        self.duration = Histogram(
            "gw_request_seconds",
            "Whole request, streams included",
            ["alias"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "gw_tokens_total", "Tokens by provider and direction", ["provider", "direction"], registry=r
        )
        self.cost = Counter("gw_cost_usd_total", "Money actually charged", ["provider", "client"], registry=r)
        self.saved = Counter(
            "gw_cost_saved_usd_total", "Money not spent thanks to the cache", ["layer"], registry=r
        )
        self.cache = Counter("gw_cache_requests_total", "Cache lookups", ["layer", "result"], registry=r)
        self.similarity = Histogram(
            "gw_cache_similarity",
            "Similarity of semantic hits",
            buckets=(0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0),
            registry=r,
        )
        self.fallbacks = Counter(
            "gw_fallbacks_total", "Requests answered by a later link", ["from", "to"], registry=r
        )
        self.shed = Counter("gw_shed_total", "Requests turned away at admission", ["priority"], registry=r)
        self.stream_outcomes = Counter(
            "gw_stream_outcomes_total", "How streams ended", ["outcome"], registry=r
        )
        self.guardrails = Counter(
            "gw_guardrail_events_total", "Guardrail decisions", ["type", "action"], registry=r
        )
        self.rate_limited = Counter(
            "gw_rate_limited_total", "Requests over the client's rpm", ["client"], registry=r
        )
        # Read at scrape time from live state:
        self.breaker = Gauge("gw_breaker_state", "0 closed, 1 half-open, 2 open", ["provider"], registry=r)
        self.in_flight = Gauge("gw_in_flight", "Requests being served", registry=r)
        self.budget_remaining = Gauge("gw_budget_remaining_usd", "Budget left", ["scope"], registry=r)
        self.budget_events = Gauge(
            "gw_budget_events", "Budget counters (reserved, rejected, late_settle, ...)", ["kind"], registry=r
        )
        self.cache_events = Gauge(
            "gw_cache_events", "Cache counters (shadow hits, writes, ...)", ["kind"], registry=r
        )

    # --- hooks --------------------------------------------------------------------------------

    def request(self, alias: str, provider: str, status: int) -> None:
        self.requests.labels(alias, provider, str(status)).inc()

    def cost_booked(self, client: str, provider: str, actual_micro: int, reference_micro: int) -> None:
        if actual_micro:
            self.cost.labels(provider, client).inc(actual_micro / 1_000_000)

    def saving(self, layer: str, micro: int) -> None:
        if micro:
            self.saved.labels(layer).inc(micro / 1_000_000)

    def usage(self, provider: str, usage: dict | None) -> None:
        if usage:
            self.tokens.labels(provider, "input").inc(int(usage.get("prompt_tokens") or 0))
            self.tokens.labels(provider, "output").inc(int(usage.get("completion_tokens") or 0))

    # --- scrape -------------------------------------------------------------------------------

    async def render(self, gw) -> bytes:
        for name, snap in gw.breakers.snapshot().items():
            self.breaker.labels(name).set(_STATE_VALUE.get(State(snap["state"]), 0))
        self.in_flight.set(gw.admission.in_flight)
        if gw.budget is not None:
            for kind, n in gw.budget.counters.items():
                self.budget_events.labels(kind).set(n)
            snap = await gw.budget.snapshot("-", None)
            for scope, v in (snap.get("scopes") or {}).items():
                if scope != "client_day" and v.get("remaining") is not None:
                    self.budget_remaining.labels(scope).set(v["remaining"] / 1_000_000)
        if gw.cache is not None:
            for kind, n in gw.cache.snapshot().get("counters", {}).items():
                self.cache_events.labels(kind).set(n)
        return generate_latest(self.registry)
