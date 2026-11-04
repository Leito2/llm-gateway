"""Concurrency limit per provider (PLAN 4.4): a slow provider fills only its own slots.

Fixed by default. A provider declared `adaptive` (AIMD, stretch) starts at its configured limit and moves it
between `min` and that maximum: +1 after a full window of successes, halved (at most once a second) when the
provider shows congestion (429, timeout, slow first token)."""

import asyncio
import time
from collections import deque
from collections.abc import Callable


class _Pool:
    def __init__(self, limit: int, adaptive: bool, minimum: int) -> None:
        self.limit, self.max, self.min, self.adaptive = limit, limit, min(minimum, limit), adaptive
        self.in_use = 0
        self.waiters: deque[asyncio.Future] = deque()
        self.successes = 0
        self.last_decrease = float("-inf")


class Bulkhead:
    def __init__(
        self,
        limits: dict[str, int],
        adaptive: dict[str, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`limits`: provider -> slots (maximum if adaptive). `adaptive`: provider -> minimum slots."""
        adaptive = adaptive or {}
        self._pools = {n: _Pool(v, n in adaptive, adaptive.get(n, v)) for n, v in limits.items()}
        self._clock = clock

    async def acquire(self, provider: str, wait_s: float) -> bool:
        """A slot, waiting up to `wait_s`; False means the provider is full and the router moves on."""
        pool = self._pools.get(provider)
        if pool is None:
            return True  # no limit declared
        deadline = self._clock() + wait_s
        while pool.in_use >= pool.limit:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return False
            waiter = asyncio.get_running_loop().create_future()
            pool.waiters.append(waiter)
            try:
                await asyncio.wait_for(waiter, timeout=remaining)
            except TimeoutError:
                with_removed = [w for w in pool.waiters if w is not waiter]
                pool.waiters = deque(with_removed)
                if pool.in_use >= pool.limit:
                    return False
        pool.in_use += 1
        return True

    def release(self, provider: str) -> None:
        pool = self._pools.get(provider)
        if pool is None:
            return
        pool.in_use -= 1
        self._wake(pool)

    @staticmethod
    def _wake(pool: _Pool) -> None:
        while pool.waiters and pool.in_use < pool.limit:
            waiter = pool.waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return

    def feedback(self, provider: str, congested: bool) -> None:
        """AIMD input from the executor. No effect on fixed providers."""
        pool = self._pools.get(provider)
        if pool is None or not pool.adaptive:
            return
        if congested:
            if self._clock() - pool.last_decrease >= 1.0:
                pool.limit = max(pool.min, pool.limit // 2)
                pool.successes = 0
                pool.last_decrease = self._clock()
            return
        pool.successes += 1
        if pool.successes >= pool.limit and pool.limit < pool.max:
            pool.limit += 1
            pool.successes = 0
            self._wake(pool)

    def snapshot(self) -> dict:
        out = {}
        for name, p in sorted(self._pools.items()):
            entry = {"limit": p.limit, "free": max(0, p.limit - p.in_use)}
            if p.adaptive:
                entry |= {"max": p.max, "min": p.min}
            out[name] = entry
        return out
