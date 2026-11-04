"""Circuit breaker per provider, with three triggers (PLAN §4.3). State lives in process memory (ADR-0003)."""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerConfig:
    consecutive_failures: int = 3  # trigger 1: failures in a row
    window: int = 20  # the last N calls feed triggers 2 and 3...
    min_calls: int = 10  # ...once at least this many exist
    error_rate: float = 0.5  # trigger 2: errors / calls above this
    slow_rate: float = 0.6  # trigger 3: slow calls / calls above this
    cooldown_s: float = 30.0  # doubles on every re-open...
    max_cooldown_s: float = 300.0  # ...up to this
    half_open_trials: int = 1  # requests let through while half-open


class CircuitBreaker:
    def __init__(
        self, config: BreakerConfig | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        config = config or BreakerConfig()
        self.config, self._clock = config, clock
        self.state = State.CLOSED
        self._calls: deque[tuple[bool, bool]] = deque(maxlen=config.window)  # (ok, slow)
        self._consecutive = 0
        self._opens = 0  # re-opens since the last close; drives the cooldown doubling
        self._open_until = 0.0
        self._trials = 0

    def peek(self) -> bool:
        """Would a call pass? Changes nothing, so the router can check before it takes other permits."""
        if self.state is State.CLOSED:
            return True
        if self.state is State.OPEN:
            return self._clock() >= self._open_until
        return self._trials < self.config.half_open_trials

    def allow(self) -> bool:
        """Ask permission for a call. Half-open lets only `half_open_trials` calls through at a time."""
        if self.state is State.OPEN and self._clock() >= self._open_until:
            self.state, self._trials = State.HALF_OPEN, 0
        if self.state is State.CLOSED:
            return True
        if self.state is State.HALF_OPEN and self._trials < self.config.half_open_trials:
            self._trials += 1
            return True
        return False

    def release_trial(self) -> None:
        """A permitted call ended without a verdict (cancelled): give the trial slot back."""
        if self.state is State.HALF_OPEN and self._trials > 0:
            self._trials -= 1

    def record_success(self, slow: bool = False) -> None:
        if self.state is State.OPEN:
            return  # a call that started before the breaker opened: not evidence
        self._calls.append((True, slow))
        self._consecutive = 0
        if self.state is State.HALF_OPEN:
            self.state, self._opens, self._trials = State.CLOSED, 0, 0
            self._calls.clear()
            return
        self._evaluate()

    def record_failure(self) -> None:
        if self.state is State.OPEN:
            return
        self._calls.append((False, False))
        self._consecutive += 1
        if self.state is State.HALF_OPEN:
            self._open()
            return
        self._evaluate()

    def _evaluate(self) -> None:
        cfg, total = self.config, len(self._calls)
        errors = sum(1 for ok, _ in self._calls if not ok)
        slow = sum(1 for _, s in self._calls if s)
        if (
            self._consecutive >= cfg.consecutive_failures
            or (total >= cfg.min_calls and errors / total > cfg.error_rate)
            or (total >= cfg.min_calls and slow / total > cfg.slow_rate)
        ):
            self._open()

    def _open(self) -> None:
        cooldown = min(self.config.cooldown_s * 2**self._opens, self.config.max_cooldown_s)
        self._opens += 1
        self.state, self._trials = State.OPEN, 0
        self._open_until = self._clock() + cooldown
        self._calls.clear()
        self._consecutive = 0

    def snapshot(self) -> dict:
        total = len(self._calls)
        return {
            "state": self.state.value,
            "consecutive_failures": self._consecutive,
            "window_calls": total,
            "error_rate": round(sum(1 for ok, _ in self._calls if not ok) / total, 2) if total else 0.0,
            "slow_rate": round(sum(1 for _, s in self._calls if s) / total, 2) if total else 0.0,
            "retry_in_s": round(max(0.0, self._open_until - self._clock()), 1)
            if self.state is State.OPEN
            else 0.0,
        }


class BreakerRegistry:
    def __init__(
        self, config: BreakerConfig | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._config, self._clock = config or BreakerConfig(), clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(self._config, self._clock)
        return self._breakers[provider]

    def snapshot(self) -> dict:
        return {name: b.snapshot() for name, b in sorted(self._breakers.items())}
