"""Fixed concurrency limit per provider (PLAN §4.4): a slow provider fills only its own slots."""

import asyncio


class Bulkhead:
    def __init__(self, limits: dict[str, int]) -> None:
        self._limits = limits
        self._slots = {name: asyncio.Semaphore(n) for name, n in limits.items()}

    async def acquire(self, provider: str, wait_s: float) -> bool:
        """A slot, waiting up to `wait_s`; False means the provider is full and the router moves on."""
        slots = self._slots.get(provider)
        if slots is None:
            return True  # no limit declared
        try:
            await asyncio.wait_for(slots.acquire(), timeout=wait_s)
        except TimeoutError:
            return False
        return True

    def release(self, provider: str) -> None:
        if provider in self._slots:
            self._slots[provider].release()

    def snapshot(self) -> dict:
        return {
            name: {"limit": n, "free": self._slots[name]._value} for name, n in sorted(self._limits.items())
        }
