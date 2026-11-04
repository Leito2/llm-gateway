"""In-process cache backend: a dict plus brute-force cosine search. Fine for one worker (ADR-0003) and a few
thousand entries; Redis replaces it when the cache must survive restarts or grow."""

import copy
import time
from collections import defaultdict
from collections.abc import Callable

from llm_gateway.cache.backend import CacheEntry, Candidate
from llm_gateway.cache.embedder import cosine


class MemoryBackend:
    def __init__(self, max_entries: int = 5000, clock: Callable[[], float] = time.time) -> None:
        self.max_entries, self._clock = max_entries, clock
        self._entries: dict[str, CacheEntry] = {}
        self._vectors: dict[str, list[float]] = {}
        self._by_ns: dict[str, set[str]] = defaultdict(set)
        self._by_tag: dict[str, set[str]] = defaultdict(set)
        self.evicted = 0

    def _live(self, key: str) -> CacheEntry | None:
        entry = self._entries.get(key)
        if entry is not None and entry.expires_at <= self._clock():
            self._drop(key)
            return None
        return entry

    def _drop(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        self._vectors.pop(key, None)
        if entry is None:
            return
        self._by_ns[entry.ns].discard(key)
        for tag in entry.tags:
            self._by_tag[tag].discard(key)

    def _make_room(self) -> None:
        """Least-hit first, oldest among equals: the same idea as Redis `volatile-lfu`."""
        while len(self._entries) >= self.max_entries:
            victim = min(self._entries.values(), key=lambda e: (e.hits, e.created_at))
            self._drop(victim.key)
            self.evicted += 1

    async def get_exact(self, key: str) -> CacheEntry | None:
        entry = self._live(key)
        return copy.deepcopy(entry) if entry else None

    async def put(self, entry: CacheEntry, vector: list[float] | None) -> None:
        self._drop(entry.key)  # a rewrite replaces the old entry and its tags
        self._make_room()
        entry = copy.deepcopy(entry)
        self._entries[entry.key] = entry
        for tag in entry.tags:
            self._by_tag[tag].add(entry.key)
        if vector is not None and entry.ns:
            self._vectors[entry.key] = vector
            self._by_ns[entry.ns].add(entry.key)

    async def search(self, ns: str, vector: list[float], k: int = 5) -> list[Candidate]:
        scored = []
        for key in list(self._by_ns.get(ns, ())):
            entry = self._live(key)
            if entry is not None:
                scored.append(Candidate(copy.deepcopy(entry), cosine(vector, self._vectors[key])))
        scored.sort(key=lambda c: c.similarity, reverse=True)
        return scored[:k]

    async def touch(self, key: str) -> None:
        if key in self._entries:
            self._entries[key].hits += 1

    async def invalidate(self, tags: set[str], client: str | None = None) -> int:
        doomed = {k for t in tags for k in self._by_tag.get(t, ())}
        doomed = {k for k in doomed if client is None or self._entries[k].client == client}
        for key in doomed:
            self._drop(key)
        return len(doomed)

    def stats(self) -> dict:
        per_ns = {ns: len(keys) for ns, keys in self._by_ns.items() if keys}
        return {
            "backend": "memory",
            "entries": len(self._entries),
            "semantic_entries": len(self._vectors),
            "namespaces": len(per_ns),
            "evicted": self.evicted,
        }
