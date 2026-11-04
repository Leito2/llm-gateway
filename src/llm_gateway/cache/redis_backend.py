"""Redis cache backend: entries survive restarts and are shared by every process (PLAN 6.6, ADR-0002).

Layout (all keys under `gwc:`):
  gwc:e:<key>    hash  d = entry as JSON, hits = counter. Has a TTL, so `volatile-lfu` may evict it.
  gwc:ns:<ns>    hash  entry key -> float32 vector (packed bytes), one hash per semantic namespace
  gwc:tag:<tag>  set   entry keys carrying that tag, for invalidation

HONEST SCOPE. PLAN names the Redis 8 Query Engine (HNSW vector search). This backend does NOT use it: the
nearest-neighbour search loads the namespace's vectors and ranks them in Python, like `MemoryBackend`, so it
works on any Redis (and on fakeredis, where it is tested) and is fine for a few thousand entries per
namespace. HNSW would replace `search` and nothing else. Eviction under memory pressure is Redis's job
(`maxmemory-policy volatile-lfu`); `max_entries` is not enforced here.

A Redis error is never the client's problem: reads become misses, writes are dropped, and `errors` counts
them. `invalidate` is the exception: it re-raises, because pretending to have invalidated is worse."""

import json
import logging
import time
from array import array
from collections.abc import Callable
from dataclasses import asdict

from redis.asyncio import Redis
from redis.exceptions import RedisError

from llm_gateway.cache.backend import CacheEntry, Candidate
from llm_gateway.cache.embedder import cosine

log = logging.getLogger("gateway.cache.redis")

ENTRY, NS, TAG = "gwc:e:", "gwc:ns:", "gwc:tag:"


def pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack(blob: bytes) -> list[float]:
    out = array("f")
    out.frombytes(blob)
    return out.tolist()


def encode(entry: CacheEntry) -> str:
    return json.dumps(asdict(entry), separators=(",", ":"))


def decode(raw: bytes | str, hits: int = 0) -> CacheEntry:
    data = json.loads(raw)
    data["tags"] = tuple(data["tags"])
    data["hits"] = hits
    return CacheEntry(**data)


class RedisBackend:
    def __init__(self, redis: Redis, clock: Callable[[], float] = time.time) -> None:
        self.redis, self._clock = redis, clock
        self.errors = 0
        self.puts = 0

    def _failed(self, what: str, exc: Exception) -> None:
        self.errors += 1
        log.warning("cache redis %s failed (%s): %s", what, type(exc).__name__, exc)

    async def _load(self, key: str) -> CacheEntry | None:
        raw, hits = await self.redis.hmget(ENTRY + key, "d", "hits")
        if raw is None:
            return None
        entry = decode(raw, int(hits or 0))
        if entry.expires_at <= self._clock():  # logical TTL; the Redis TTL is only a safety net for eviction
            await self._forget(entry)
            return None
        return entry

    async def _forget(self, entry: CacheEntry) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(ENTRY + entry.key)
            if entry.ns:
                pipe.hdel(NS + entry.ns, entry.key)
            for tag in entry.tags:
                pipe.srem(TAG + tag, entry.key)
            await pipe.execute()

    async def get_exact(self, key: str) -> CacheEntry | None:
        try:
            return await self._load(key)
        except RedisError as exc:
            self._failed("get", exc)
            return None

    async def put(self, entry: CacheEntry, vector: list[float] | None) -> None:
        try:
            old = await self._load(entry.key)
            ttl = max(1, int(entry.expires_at - self._clock()))
            async with self.redis.pipeline(transaction=True) as pipe:
                if old is not None:  # a rewrite replaces the old entry's index rows
                    if old.ns:
                        pipe.hdel(NS + old.ns, old.key)
                    for tag in old.tags:
                        pipe.srem(TAG + tag, old.key)
                pipe.hset(ENTRY + entry.key, mapping={"d": encode(entry), "hits": 0})
                pipe.expire(ENTRY + entry.key, ttl)
                if vector is not None and entry.ns:
                    pipe.hset(NS + entry.ns, entry.key, pack(vector))
                for tag in entry.tags:
                    pipe.sadd(TAG + tag, entry.key)
                await pipe.execute()
            self.puts += 1
        except RedisError as exc:
            self._failed("put", exc)

    async def search(self, ns: str, vector: list[float], k: int = 5) -> list[Candidate]:
        try:
            stored = await self.redis.hgetall(NS + ns)
            ranked = sorted(
                ((cosine(vector, unpack(blob)), key.decode()) for key, blob in stored.items()), reverse=True
            )
            found: list[Candidate] = []
            for similarity, key in ranked:
                if len(found) == k:
                    break
                entry = await self._load(key)
                if entry is None:  # expired or evicted: the vector row is stale
                    await self.redis.hdel(NS + ns, key)
                    continue
                found.append(Candidate(entry, similarity))
            return found
        except RedisError as exc:
            self._failed("search", exc)
            return []

    async def touch(self, key: str) -> None:
        try:
            await self.redis.hincrby(ENTRY + key, "hits", 1)
        except RedisError as exc:
            self._failed("touch", exc)

    async def invalidate(self, tags: set[str], client: str | None = None) -> int:
        removed = 0
        keys: set[str] = set()
        for tag in tags:
            keys |= {k.decode() for k in await self.redis.smembers(TAG + tag)}
        for key in keys:
            entry = await self._load(key)
            if entry is None:
                for tag in tags:  # expired or evicted: drop the stale index row
                    await self.redis.srem(TAG + tag, key)
                continue
            if client is None or entry.client == client:
                await self._forget(entry)
                removed += 1
        return removed

    def stats(self) -> dict:
        return {"backend": "redis", "puts": self.puts, "errors": self.errors}
