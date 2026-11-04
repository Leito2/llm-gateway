"""Runs one request down its fallback chain (PLAN §4): gates, timeouts per phase, retries, breaker accounting.

Gates, in order, per link: breaker (peek) -> outbound quota -> bulkhead slot -> breaker (take permit).
Retry and fallback happen only before the first token leaves (ADR-4), so a stream is probed by taking its
first chunk here. After that the caller owns the stream and the `Lease` that holds the bulkhead slot."""

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.providers.base import Provider, ProviderError
from llm_gateway.quotas.limiter import QuotaLimiter
from llm_gateway.resilience.breaker import BreakerRegistry, CircuitBreaker, State
from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.resilience.retry import backoff_s
from llm_gateway.routing.config import AliasPolicy, Target

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # provider calls per request, retries included (PLAN §4.2)
MAX_RETRIES_PER_LINK = 1
SLOW_FRACTION = 0.8  # a first token later than 80% of the TTFT timeout counts as a slow call
BULKHEAD_WAIT_S = 0.05  # a full provider is skipped, not queued behind
BLOCK_AFTER_429_S = 30.0  # when the provider sends a 429 without Retry-After


class Lease:
    """Holds a provider's bulkhead slot for as long as a call or stream is alive. `close` is idempotent."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release

    def close(self) -> None:
        release, self._release = self._release, lambda: None
        release()


@dataclass
class Attempt:
    """The link that answered. Non-streaming: `completion`. Streaming: `first` + `stream` + a live `lease`."""

    target: Target
    depth: int  # links tried so far, this one included (X-Gateway-Fallback-Depth)
    breaker: CircuitBreaker
    lease: Lease
    deadline: float  # absolute time on the executor's clock; total_timeout always wins
    completion: dict | None = None
    first: dict | None = None
    stream: AsyncIterator[dict] | None = None


class ChainFailed(Exception):
    def __init__(
        self, *, last_error: ProviderError | None, tried: int, timed_out: bool, skipped: dict[str, int]
    ) -> None:
        super().__init__("no link could answer")
        self.last_error, self.tried, self.timed_out, self.skipped = last_error, tried, timed_out, skipped


@dataclass
class Executor:
    providers: dict[str, Provider]
    breakers: BreakerRegistry
    bulkhead: Bulkhead
    quotas: QuotaLimiter
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    rng: random.Random = field(default_factory=random.Random)

    async def run(
        self,
        *,
        policy: AliasPolicy,
        chain: list[Target],
        request: ChatCompletionRequest,
        quota_tokens: int,
        stream: bool,
    ) -> Attempt:
        """First link that answers. Raises ProviderError(kind="client") as is, ChainFailed otherwise."""
        deadline = self.clock() + policy.total_timeout_s
        attempts = tried = 0
        last_error: ProviderError | None = None
        skipped = {"breaker": 0, "quota": 0, "bulkhead": 0}

        for target in chain:
            if attempts >= MAX_ATTEMPTS:
                break
            name = target.provider
            breaker = self.breakers.get(name)
            if not breaker.peek():
                skipped["breaker"] += 1
                continue
            if not self.quotas.try_acquire(name, quota_tokens):
                skipped["quota"] += 1
                continue
            if not await self.bulkhead.acquire(name, BULKHEAD_WAIT_S):
                skipped["bulkhead"] += 1
                continue
            if not breaker.allow():  # another request took the half-open trial meanwhile
                self.bulkhead.release(name)
                skipped["breaker"] += 1
                continue

            tried += 1
            lease = Lease(lambda n=name: self.bulkhead.release(n))
            retries = 0
            while True:
                if self.clock() >= deadline:
                    lease.close()
                    breaker.release_trial()
                    raise ChainFailed(last_error=last_error, tried=tried, timed_out=True, skipped=skipped)
                attempts += 1
                try:
                    return await self._call(target, breaker, lease, tried, deadline, policy, request, stream)
                except ProviderError as exc:
                    exc.provider = name
                    if exc.kind == "client":
                        lease.close()
                        breaker.release_trial()
                        raise
                    last_error = exc
                    breaker.record_failure()
                    if exc.status == 429:
                        self.quotas.block(name, exc.retry_after or BLOCK_AFTER_429_S)
                    log.warning("provider %s failed (%s): %s", name, exc.kind, exc)
                    if not self._may_retry(exc, breaker, retries, attempts, deadline):
                        break
                    retries += 1
                    await self.sleep(backoff_s(retries - 1, rng=self.rng))
                except BaseException:  # cancelled (client left): no verdict on the provider
                    lease.close()
                    breaker.release_trial()
                    raise
            lease.close()

        raise ChainFailed(
            last_error=last_error, tried=tried, timed_out=self.clock() >= deadline, skipped=skipped
        )

    def _may_retry(
        self, exc: ProviderError, breaker: CircuitBreaker, retries: int, attempts: int, deadline: float
    ) -> bool:
        """Connection errors and 5xx only: a timeout, a 429 or a 4xx would fail the same way again."""
        return (
            exc.kind == "transient"
            and (exc.status is None or exc.status >= 500)
            and retries < MAX_RETRIES_PER_LINK
            and attempts < MAX_ATTEMPTS
            and breaker.state is State.CLOSED
            and self.clock() < deadline
        )

    async def _call(
        self,
        target: Target,
        breaker: CircuitBreaker,
        lease: Lease,
        depth: int,
        deadline: float,
        policy: AliasPolicy,
        request: ChatCompletionRequest,
        stream: bool,
    ) -> Attempt:
        try:
            return await self._call_once(target, breaker, lease, depth, deadline, policy, request, stream)
        except (ProviderError, asyncio.CancelledError):
            raise
        except Exception as exc:  # an adapter bug is one failed link, not a crashed request
            log.exception("provider %s raised an unexpected error", target.provider)
            raise ProviderError(
                f"{target.provider}: unexpected {type(exc).__name__}", kind="provider"
            ) from exc

    async def _call_once(
        self,
        target: Target,
        breaker: CircuitBreaker,
        lease: Lease,
        depth: int,
        deadline: float,
        policy: AliasPolicy,
        request: ChatCompletionRequest,
        stream: bool,
    ) -> Attempt:
        provider = self.providers[target.provider]
        started = self.clock()
        remaining = max(0.0, deadline - started)
        if not stream:
            # Without streaming there is no first token to time: the whole call is bounded by total_timeout.
            try:
                async with asyncio.timeout(remaining):
                    completion = await provider.complete(request, target.model)
            except TimeoutError as exc:
                raise ProviderError(
                    f"{target.provider}: no answer within {remaining:.1f}s", kind="transient", status=408
                ) from exc
            breaker.record_success()
            lease.close()
            return Attempt(target, depth, breaker, lease, deadline, completion=completion)

        gen = provider.stream(request, target.model)
        try:
            async with asyncio.timeout(min(policy.ttft_timeout_s, remaining)):
                first = await anext(gen)
        except TimeoutError as exc:
            await _aclose(gen)
            raise ProviderError(
                f"{target.provider}: no first token within {policy.ttft_timeout_s:g}s",
                kind="transient",
                status=408,  # 408: a timeout is never retried on the same link
            ) from exc
        except StopAsyncIteration as exc:
            raise ProviderError(f"{target.provider}: empty stream", kind="provider") from exc
        except BaseException:
            await _aclose(gen)
            raise
        breaker.record_success(slow=self.clock() - started > SLOW_FRACTION * policy.ttft_timeout_s)
        return Attempt(target, depth, breaker, lease, deadline, first=first, stream=gen)


async def _aclose(gen: AsyncIterator[dict]) -> None:
    aclose = getattr(gen, "aclose", None)
    if aclose:
        await aclose()
