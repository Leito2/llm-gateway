"""The cache as the chat handler sees it: look up before the providers, write back after (PLAN §6).

Lookup order: exact key (L1), then semantic search (L2: namespace → embedding → KNN → thresholds → lexical
guards → verifier in the gray zone). Every failure of the semantic path (no embedder, timeout, error) is a
miss, never an error: the request goes on to the providers. Write-back runs off the request path."""

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.cache import guards
from llm_gateway.cache.backend import CacheBackend, CacheEntry
from llm_gateway.cache.config import CacheConfig
from llm_gateway.cache.embedder import Embedder
from llm_gateway.cache.keys import (
    exact_key,
    has_non_text,
    namespace_base,
    parse_tags,
    semantic_namespace,
    semantic_text,
    system_hash,
)
from llm_gateway.routing.config import AliasPolicy

log = logging.getLogger(__name__)

Layer = Literal["exact", "semantic", "verified"]


class Verifier(Protocol):
    """Cross-encoder for the gray zone (MiniLM in ONNX, PLAN §6.3 point 3)."""

    async def equivalent(self, new: str, cached: str) -> bool: ...


@dataclass
class CacheRequest:
    client: str
    alias: str
    key: str  # exact key
    base: str  # namespace base (no embedder)
    sem_text: str | None  # None: no semantic layer for this request
    tags: tuple[str, ...]
    ttl_s: int


@dataclass
class Hit:
    layer: Layer
    entry: CacheEntry
    similarity: float | None = None


class CacheService:
    def __init__(
        self,
        backend: CacheBackend,
        config: CacheConfig,
        embedder: Embedder | None = None,
        verifier: Verifier | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.backend, self.config, self.embedder, self.verifier = backend, config, embedder, verifier
        self._clock = clock
        self._pending: set[asyncio.Task] = set()
        self.counters: Counter[str] = Counter()
        self.guard_rejections: Counter[str] = Counter()
        self.observed: list[dict] = []  # the last few "would have served" candidates
        self._observed_max = 50

    # --- request side -----------------------------------------------------------------------------------

    def prepare(
        self,
        client: str,
        alias: str,
        policy: AliasPolicy,
        body: ChatCompletionRequest,
        headers: dict[str, str],
    ) -> CacheRequest | None:
        """None when this request must not touch the cache at all (R21, opt-outs, nothing enabled)."""
        if body.cache is False or headers.get("x-cache-scope", "").strip().lower() == "private":
            return None
        if has_non_text(body.messages):
            return None
        exact = policy.cache or policy.semantic_cache if body.cache is None else True
        if not exact:
            return None
        rf = body.response_format.model_dump(exclude_none=True) if body.response_format else None
        base = namespace_base(
            client, alias, system_hash(body.messages), headers.get("x-prompt-version", ""), rf
        )
        sem = None
        if policy.semantic_cache and self.embedder is not None:
            sem = semantic_text(body.messages, policy.semantic_scope)
        return CacheRequest(
            client,
            alias,
            exact_key(base, body.messages),
            base,
            sem,
            parse_tags(headers.get("x-cache-tags")),
            policy.cache_ttl_s,
        )

    async def lookup(self, req: CacheRequest) -> Hit | None:
        entry = await self.backend.get_exact(req.key)
        if entry is not None:
            await self.backend.touch(req.key)
            self.counters["hit_exact"] += 1
            return Hit("exact", entry)
        hit = await self._semantic(req) if req.sem_text else None
        if hit is None:
            self.counters["miss"] += 1
            return None
        await self.backend.touch(hit.entry.key)
        self.counters[f"hit_{hit.layer}"] += 1
        return hit

    async def _semantic(self, req: CacheRequest) -> Hit | None:
        assert self.embedder is not None and req.sem_text is not None
        try:
            async with asyncio.timeout(self.config.embed_timeout_s):
                (vector,) = await self.embedder.embed([req.sem_text])
            candidates = await self.backend.search(semantic_namespace(req.base, self.embedder.name), vector)
        except Exception as exc:  # TimeoutError included: fall back to exact only
            self.counters["semantic_error"] += 1
            log.warning("semantic lookup skipped (%s): %s", type(exc).__name__, exc)
            return None
        if not candidates:
            return None
        limits = self.config.thresholds(self.embedder.name)
        if not limits.calibrated:
            best = candidates[0]
            self._observe(req, best.similarity, best.entry, guards.mismatch(req.sem_text, best.entry.text))
            return None
        for cand in candidates:
            if cand.similarity < limits.t_low:
                break  # sorted: nothing below can qualify
            reason = guards.mismatch(req.sem_text, cand.entry.text)
            if reason:
                self.guard_rejections[reason] += 1
                continue
            if cand.similarity >= limits.t_high:
                return Hit("semantic", cand.entry, cand.similarity)
            if await self._verified(req.sem_text, cand.entry.text):
                return Hit("verified", cand.entry, cand.similarity)
            self.counters["verifier_rejected"] += 1
        return None

    async def _verified(self, new: str, cached: str) -> bool:
        if self.verifier is None:
            return False
        try:
            async with asyncio.timeout(self.config.verify_timeout_s):
                return await self.verifier.equivalent(new, cached)
        except Exception as exc:
            log.warning("verifier failed (%s): treating as a miss", type(exc).__name__)
            return False

    def _observe(self, req: CacheRequest, similarity: float, entry: CacheEntry, guard: str | None) -> None:
        """Observation mode: record what a calibrated cache would have looked at, serve nothing."""
        self.counters["observed"] += 1
        self.observed.append(
            {
                "alias": req.alias,
                "similarity": round(similarity, 4),
                "guard": guard,
                "question": req.sem_text[:120],
                "candidate": entry.text[:120],
            }
        )
        del self.observed[: -self._observed_max]

    # --- write-back ---------------------------------------------------------------------------------------

    def store_later(self, req: CacheRequest, completion: dict, provider: str, *, degraded: bool) -> bool:
        """Queue a write if the answer is safe to reuse (PLAN §6.2). True when queued."""
        choice = (completion.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content")
        if degraded or choice.get("finish_reason") != "stop" or not text:
            self.counters["store_skipped"] += 1
            return False
        if len(self._pending) >= self.config.write_queue:
            self.counters["store_dropped"] += 1
            return False
        task = asyncio.get_running_loop().create_task(self._write(req, completion, provider))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return True

    async def _write(self, req: CacheRequest, completion: dict, provider: str) -> None:
        try:
            vector, ns = None, ""
            if req.sem_text and self.embedder is not None:
                try:
                    async with asyncio.timeout(self.config.embed_timeout_s * 3):
                        (vector,) = await self.embedder.embed([req.sem_text])
                    ns = semantic_namespace(req.base, self.embedder.name)
                except Exception as exc:  # store it exact-only rather than lose it
                    log.warning("embedding for write-back failed (%s)", type(exc).__name__)
            now = self._clock()
            await self.backend.put(
                CacheEntry(
                    req.key,
                    ns,
                    req.client,
                    req.alias,
                    completion,
                    provider,
                    req.sem_text or "",
                    req.tags,
                    now,
                    now + req.ttl_s,
                ),
                vector,
            )
            self.counters["stored"] += 1
        except Exception:
            self.counters["store_error"] += 1
            log.exception("cache write-back failed")

    async def flush(self, timeout_s: float = 2.0) -> None:
        """Wait for queued writes (shutdown, tests)."""
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=timeout_s)

    # --- admin ----------------------------------------------------------------------------------------------

    async def invalidate(self, tags: set[str], client: str | None = None) -> int:
        removed = await self.backend.invalidate(tags, client)
        self.counters["invalidated"] += removed
        return removed

    def snapshot(self) -> dict:
        limits = self.config.thresholds(self.embedder.name) if self.embedder else None
        if self.embedder is None:
            semantic = "off (no embedder)"
        else:
            semantic = "serving" if limits and limits.calibrated else "observing (thresholds not calibrated)"
        return {
            **self.backend.stats(),
            "embedder": self.embedder.name if self.embedder else None,
            "semantic": semantic,
            "counters": dict(self.counters),
            "guard_rejections": dict(self.guard_rejections),
            "pending_writes": len(self._pending),
            "recent_observed": self.observed[-5:],
        }
