"""In-process budget store with the same semantics as the Redis one (dev, tests, single worker)."""

import time
from collections import defaultdict
from collections.abc import Callable

from llm_gateway.budget.store import Caps, Scopes


class MemoryBudget:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._spent: dict[str, int] = defaultdict(int)
        self._reserved: dict[str, int] = defaultdict(int)
        self._res: dict[str, tuple[float, int, list[str]]] = {}  # id -> (deadline, amount, scope keys)

    def _release_expired(self) -> None:
        now = self._clock()
        for rid in [r for r, (deadline, _, _) in self._res.items() if deadline <= now]:
            _, amount, keys = self._res.pop(rid)
            for k in keys:
                self._reserved[k] -= amount

    async def reserve(self, rid: str, amount: int, ttl_s: float, scopes: Scopes, caps: Caps) -> bool:
        self._release_expired()
        pairs = zip(scopes.as_list(), (caps.run, caps.client_day, caps.global_day), strict=True)
        if any(cap is not None and self._spent[k] + self._reserved[k] + amount > cap for k, cap in pairs):
            return False
        for k in scopes.as_list():
            self._reserved[k] += amount
        self._res[rid] = (self._clock() + ttl_s, amount, scopes.as_list())
        return True

    async def settle(self, rid: str, real: int, scopes: Scopes) -> bool:
        held = self._res.pop(rid, None)
        if held is not None:
            for k in held[2]:
                self._reserved[k] -= held[1]
        for k in scopes.as_list():
            self._spent[k] += real  # even when late: recorded spend is never below real spend
        return held is None

    async def snapshot(self, scopes: Scopes) -> dict:
        self._release_expired()
        return {
            name: {"spent": self._spent[k], "reserved": self._reserved[k]}
            for name, k in zip(("run", "client_day", "global_day"), scopes.as_list(), strict=True)
        }
