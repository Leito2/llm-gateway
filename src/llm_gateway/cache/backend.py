"""Storage and search behind one interface (PLAN §6.6, ADR-0002). `MemoryBackend` is the default for dev and
tests; the Redis 8 Query Engine backend (`redis_local`) implements the same Protocol."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class CacheEntry:
    key: str  # exact key: namespace base + the whole conversation
    ns: str  # semantic namespace (base × embedder); "" when the entry has no vector
    client: str
    alias: str
    response: dict  # the `chat.completion` that gets served
    provider: str  # "provider:model" that really produced it (X-Gateway-Provider on a hit)
    text: str  # the embedded question, for guards and the verifier
    tags: tuple[str, ...]
    created_at: float
    expires_at: float
    hits: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Candidate:
    entry: CacheEntry
    similarity: float


class CacheBackend(Protocol):
    async def get_exact(self, key: str) -> CacheEntry | None: ...

    async def put(self, entry: CacheEntry, vector: list[float] | None) -> None: ...

    async def search(self, ns: str, vector: list[float], k: int = 5) -> list[Candidate]:
        """The k nearest live entries of `ns`, best first."""
        ...

    async def touch(self, key: str) -> None:
        """Count a hit (feeds eviction)."""
        ...

    async def invalidate(self, tags: set[str], client: str | None = None) -> int:
        """Drop every entry that has any of `tags` (only the client's own when `client` is given)."""
        ...

    def stats(self) -> dict: ...
