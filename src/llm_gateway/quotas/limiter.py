"""Token buckets per provider. In memory for now: the gateway runs one worker (ADR-0003). Redis-backed
buckets, which survive restarts, arrive with the budget in M5."""

import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class QuotaSpec(BaseModel):
    rpm: int | None = Field(default=None, gt=0)  # requests per minute
    tpm: int | None = Field(default=None, gt=0)  # tokens per minute
    rpd: int | None = Field(default=None, gt=0)  # requests per day (UTC)


class _Bucket:
    """Holds up to `capacity`, refills evenly over one minute."""

    def __init__(self, capacity: int, now: float) -> None:
        self.capacity, self.level, self.updated = capacity, float(capacity), now

    def refill(self, now: float) -> None:
        self.level = min(self.capacity, self.level + (now - self.updated) * self.capacity / 60.0)
        self.updated = now


class QuotaLimiter:
    def __init__(
        self,
        specs: dict[str, QuotaSpec],
        *,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] = lambda: datetime.now(UTC).date().isoformat(),
    ) -> None:
        self._specs, self._clock, self._today = specs, clock, today
        self._rpm = {p: _Bucket(s.rpm, clock()) for p, s in specs.items() if s.rpm}
        self._tpm = {p: _Bucket(s.tpm, clock()) for p, s in specs.items() if s.tpm}
        self._day: dict[str, tuple[str, int]] = {}  # provider -> (date, requests today)
        self._blocked_until: dict[str, float] = {}

    def try_acquire(self, provider: str, tokens: int) -> bool:
        """Take one request and `tokens` tokens, or take nothing and return False."""
        now = self._clock()
        if self._blocked_until.get(provider, 0.0) > now:
            return False
        rpm, tpm, spec = self._rpm.get(provider), self._tpm.get(provider), self._specs.get(provider)
        need = 0
        if tpm:
            tpm.refill(now)
            need = min(tokens, tpm.capacity)  # a prompt above the whole quota passes only when full
        if rpm:
            rpm.refill(now)
        day, used = self._day.get(provider, (self._today(), 0))
        if day != self._today():
            day, used = self._today(), 0
        if (rpm and rpm.level < 1) or (tpm and tpm.level < need) or (spec and spec.rpd and used >= spec.rpd):
            return False
        if rpm:
            rpm.level -= 1
        if tpm:
            tpm.level -= need
        self._day[provider] = (day, used + 1)
        return True

    def block(self, provider: str, seconds: float) -> None:
        """The provider said 429: the real quota is lower than configured. Stay away for a while."""
        self._blocked_until[provider] = self._clock() + seconds
