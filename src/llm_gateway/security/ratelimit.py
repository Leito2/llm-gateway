"""Per-client requests-per-minute limit (PLAN §8.3, M8).

Sliding window in memory (exact, fine for the single worker of ADR-0003). With Redis the count is shared in a
fixed one-minute window; if Redis fails the limiter silently falls back to the in-memory window."""

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable

from redis.exceptions import RedisError

log = logging.getLogger(__name__)
WINDOW_S = 60.0

_INCR = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return n
"""


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic, redis=None) -> None:
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._redis = redis
        self._incr = redis.register_script(_INCR) if redis is not None else None
        self.fallbacks = 0

    async def check(self, client: str, rpm: int) -> tuple[bool, float]:
        """(allowed, seconds until a slot frees up)."""
        if self._incr is not None:
            try:
                window = int(time.time() // WINDOW_S)
                n = int(await self._incr(keys=[f"ratelimit:{client}:{window}"], args=[int(WINDOW_S) + 1]))
                if n <= rpm:
                    return True, 0.0
                return False, max(1.0, (window + 1) * WINDOW_S - time.time())
            except RedisError:
                self.fallbacks += 1
                log.warning("rate limiter: Redis unavailable, using the in-memory window")
        return self._check_memory(client, rpm)

    def _check_memory(self, client: str, rpm: int) -> tuple[bool, float]:
        now, hits = self._clock(), self._hits[client]
        while hits and hits[0] <= now - WINDOW_S:
            hits.popleft()
        if len(hits) >= rpm:
            return False, max(1.0, hits[0] + WINDOW_S - now)
        hits.append(now)
        return True, 0.0
