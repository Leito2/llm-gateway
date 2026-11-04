"""Singleflight (PLAN §4.7): identical cacheable requests in flight share one provider call."""

import asyncio
import hashlib
import json
from dataclasses import dataclass


@dataclass
class Shared:
    completion: dict
    provider: str  # "provider:model" that answered, for X-Gateway-Provider
    depth: int = 1  # X-Gateway-Fallback-Depth of the leader


class Flight:
    def __init__(self, table: dict[str, "Flight"], key: str) -> None:
        self._table, self._key = table, key
        self._future: asyncio.Future[Shared | None] = asyncio.get_running_loop().create_future()

    def resolve(self, shared: Shared | None) -> None:
        """Leader only. None means the leader failed: followers then run on their own. Idempotent."""
        if self._table.get(self._key) is self:
            del self._table[self._key]
        if not self._future.done():
            self._future.set_result(shared)

    async def wait(self, wait_s: float) -> Shared | None:
        """Follower side. A follower that leaves (disconnect, timeout) must not cancel the leader's call."""
        try:
            return await asyncio.wait_for(asyncio.shield(self._future), timeout=wait_s)
        except TimeoutError:
            return None


class Singleflight:
    def __init__(self) -> None:
        self._flights: dict[str, Flight] = {}

    def join(self, key: str) -> tuple[bool, Flight]:
        """(True, flight): the caller leads and must resolve it. (False, flight): wait for it."""
        existing = self._flights.get(key)
        if existing is not None:
            return False, existing
        flight = self._flights[key] = Flight(self._flights, key)
        return True, flight

    @property
    def in_flight(self) -> int:
        return len(self._flights)


def flight_key(
    client: str,
    alias: str,
    messages: list[dict],
    response_format: dict | None,
    max_tokens: int,
    temperature: float,
) -> str:
    """Same client, alias and parameters: requests that would get the same answer from the same provider."""
    payload = json.dumps(
        [client, alias, messages, response_format, max_tokens, temperature],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
