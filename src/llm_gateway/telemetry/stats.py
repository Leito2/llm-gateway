"""In-memory request counters behind GET /stats (Prometheus metrics replace this in M8)."""

import time
from collections import Counter


class Stats:
    def __init__(self, metrics=None) -> None:
        self.metrics = metrics  # telemetry.metrics.Metrics, fed by every record below
        self.started = time.monotonic()
        self._counts: Counter[tuple[str, str, int]] = Counter()
        self._cost: Counter[tuple[str, str]] = Counter()  # (client, provider) -> micro-dollars charged
        self._reference: Counter[tuple[str, str]] = Counter()  # same key, at paid reference prices
        self._saved: Counter[str] = Counter()  # cache layer -> micro-dollars not spent

    def record(self, alias: str, provider: str, status: int) -> None:
        self._counts[(alias, provider, status)] += 1
        if self.metrics:
            self.metrics.request(alias, provider, status)

    def record_cost(self, client: str, provider: str, actual: int, reference: int) -> None:
        self._cost[(client, provider)] += actual
        self._reference[(client, provider)] += reference
        if self.metrics:
            self.metrics.cost_booked(client, provider, actual, reference)

    def record_saving(self, layer: str, micro: int) -> None:
        self._saved[layer] += micro
        if self.metrics:
            self.metrics.saving(layer, micro)

    def snapshot(self) -> dict:
        return {
            "cost_micro_usd": [
                {"client": c, "provider": p, "actual": n, "reference": self._reference[(c, p)]}
                for (c, p), n in sorted(self._cost.items())
            ],
            "cache_saved_micro_usd": dict(self._saved),
            "uptime_s": round(time.monotonic() - self.started, 1),
            "requests_total": sum(self._counts.values()),
            "requests": [
                {"alias": a, "provider": p, "status": s, "count": n}
                for (a, p, s), n in sorted(self._counts.items())
            ],
        }
