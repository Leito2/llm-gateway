"""Per-provider latency memory (stretch): an EWMA of time-to-first-token that feeds two decisions.

1. `reorder: ewma` on an alias sorts its non-degraded links fastest first (links nobody has measured yet go
   first so they get measured), degraded links stay last.
2. Hedging derives its delay from it: wait about three typical latencies before racing the next link."""

from collections import defaultdict

from llm_gateway.routing.config import Target

ALPHA = 0.2
MIN_SAMPLES_FOR_HEDGE = 5


class LatencyTracker:
    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = alpha
        self._ewma: dict[str, float] = {}
        self._samples: dict[str, int] = defaultdict(int)

    def record(self, provider: str, seconds: float) -> None:
        old = self._ewma.get(provider)
        self._ewma[provider] = seconds if old is None else old + self.alpha * (seconds - old)
        self._samples[provider] += 1

    def ewma(self, provider: str) -> float | None:
        return self._ewma.get(provider)

    def samples(self, provider: str) -> int:
        return self._samples[provider]

    def reorder(self, chain: list[Target]) -> list[Target]:
        good = sorted((t for t in chain if not t.degraded), key=lambda t: self._ewma.get(t.provider, 0.0))
        return [*good, *(t for t in chain if t.degraded)]

    def snapshot(self) -> dict:
        return {
            p: {"ewma_ttft_s": round(v, 4), "samples": self._samples[p]}
            for p, v in sorted(self._ewma.items())
        }
