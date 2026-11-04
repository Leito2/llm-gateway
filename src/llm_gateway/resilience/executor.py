"""Runs one request down its fallback chain (PLAN §4): gates, timeouts per phase, retries, breaker accounting.

Gates, in order, per link: breaker (peek) -> outbound quota -> bulkhead slot -> breaker (take permit).
Retry and fallback happen only before the first token leaves (ADR-4), so a stream is probed by taking its
first chunk here. After that the caller owns the stream and the `Lease` that holds the bulkhead slot."""

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.budget.service import BudgetGate, BudgetSkip
from llm_gateway.budget.store import Reservation
from llm_gateway.providers.base import Provider, ProviderError
from llm_gateway.quotas.limiter import QuotaLimiter
from llm_gateway.resilience.breaker import BreakerRegistry, CircuitBreaker, State
from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.resilience.retry import backoff_s
from llm_gateway.routing.config import AliasPolicy, Target
from llm_gateway.routing.latency import MIN_SAMPLES_FOR_HEDGE, LatencyTracker
from llm_gateway.structured.validate import OutputValidator, merge_usage, repair_request
from llm_gateway.telemetry.tracing import Tracer

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
    gate: BudgetGate | None = None
    reservation: Reservation | None = None  # streams: the caller settles it when the stream ends
    extra_calls: int = 0  # provider calls beyond the first (a structured-output repair)
    hedge: str | None = None  # 'won' when the hedged link answered first, 'primary' when the first link did


class InvalidOutput(ProviderError):
    """The link answered, but not in the requested `response_format`, even after one repair (R15).
    A model quality problem, not an availability one: the breaker does not count it."""

    def __init__(self, message: str, *, usage: dict | None, calls: int) -> None:
        super().__init__(message, kind="provider")
        self.usage, self.calls = usage, calls  # what the failed calls cost, and how many were made


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
    tracer: Tracer | None = None  # one span per provider attempt, when tracing is on
    latency: LatencyTracker | None = None  # EWMA of time to first token per provider
    on_hedge: Callable[[str], None] | None = None  # 'launched' | 'won' | 'lost', for metrics

    async def run(
        self,
        *,
        policy: AliasPolicy,
        chain: list[Target],
        request: ChatCompletionRequest,
        quota_tokens: int,
        stream: bool,
        gate: BudgetGate | None = None,
        validate: OutputValidator | None = None,
    ) -> Attempt:
        """First link that answers. Raises ProviderError(kind="client") as is, ChainFailed otherwise."""
        kw = dict(
            policy=policy, request=request, quota_tokens=quota_tokens, stream=stream, gate=gate,
            validate=validate, deadline=self.clock() + policy.total_timeout_s,
        )  # fmt: skip
        delay = self._hedge_delay(policy, chain) if len(chain) > 1 else None
        if delay is None:
            return await self._run(chain=chain, **kw)
        return await self._run_hedged(delay, chain, kw)

    # --- hedging (stretch, ADR-12) --------------------------------------------------------------

    def _hedge_delay(self, policy: AliasPolicy, chain: list[Target]) -> float | None:
        """How long the first link may stay silent before the next one is raced. None: do not hedge."""
        if not policy.hedge:
            return None
        if policy.hedge_delay_s is not None:
            return policy.hedge_delay_s
        tracker = self.latency
        first = chain[0].provider
        if tracker is None or tracker.samples(first) < MIN_SAMPLES_FOR_HEDGE:
            return None  # not enough history to know what "slow" means for this provider
        typical = tracker.ewma(first) or 0.0
        return min(max(3 * typical, 0.05), 0.8 * policy.ttft_timeout_s)

    async def _run_hedged(self, delay: float, chain: list[Target], kw: dict) -> Attempt:
        """First link alone for `delay`; if it is still silent, race the second. First answer wins and the
        loser is cancelled. Both are tried only up to the first token (ADR-4), so nothing was emitted yet."""
        primary = asyncio.ensure_future(self._run(chain=chain[:1], **kw))
        done, _ = await asyncio.wait({primary}, timeout=delay)
        failures: list[ChainFailed] = []
        if done:
            try:
                return primary.result()
            except ChainFailed as exc:  # failed fast: ordinary fallback, no race
                return await self._continue(chain, 1, [exc], kw)
        hedge = asyncio.ensure_future(self._run(chain=chain[1:2], depth_offset=1, **kw))
        self._notify("launched")
        racers = {primary, hedge}
        try:
            while racers:
                finished, racers = await asyncio.wait(racers, return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    exc = task.exception()
                    if exc is None:
                        winner = task.result()
                        await self._cancel_all(racers)
                        for other in finished - {task}:  # both finished in the same tick: drop the loser
                            if other.exception() is None:
                                await self._discard(other.result())
                        if task is hedge:
                            winner.hedge = "won"
                            self.breakers.get(chain[0].provider).record_success(slow=True)
                            self._notify("won")
                        else:
                            winner.hedge = "primary"
                            self._notify("lost")
                        return winner
                    if isinstance(exc, ChainFailed):
                        failures.append(exc)
                    else:  # a client error or a bug: no point in waiting for the other link
                        await self._cancel_all(racers)
                        raise exc
        except BaseException:
            await self._cancel_all({primary, hedge})
            raise
        return await self._continue(chain, 2, failures, kw)

    async def _continue(
        self, chain: list[Target], start: int, failures: list[ChainFailed], kw: dict
    ) -> Attempt:
        """The links not yet tried, one at a time. Earlier errors are merged into the final failure."""
        try:
            return await self._run(chain=chain[start:], depth_offset=start, **kw)
        except ChainFailed as last:
            merged = {
                k: last.skipped.get(k, 0) + sum(f.skipped.get(k, 0) for f in failures) for k in last.skipped
            }
            raise ChainFailed(
                last_error=last.last_error
                or next((f.last_error for f in reversed(failures) if f.last_error), None),
                tried=last.tried + sum(f.tried for f in failures),
                timed_out=last.timed_out or any(f.timed_out for f in failures),
                skipped=merged,
            ) from last

    def _notify(self, outcome: str) -> None:
        if self.on_hedge:
            self.on_hedge(outcome)

    @staticmethod
    async def _cancel_all(tasks: set[asyncio.Future]) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _discard(attempt: Attempt) -> None:
        """An answer nobody will use (the race had two finishers): release everything it holds."""
        if attempt.stream is not None:
            await _aclose(attempt.stream)
        attempt.lease.close()
        if attempt.gate is not None and attempt.reservation is not None:
            await attempt.gate.settle(attempt.reservation, None)

    async def _run(
        self,
        *,
        policy: AliasPolicy,
        chain: list[Target],
        request: ChatCompletionRequest,
        quota_tokens: int,
        stream: bool,
        gate: BudgetGate | None,
        validate: OutputValidator | None,
        deadline: float,
        depth_offset: int = 0,
    ) -> Attempt:
        attempts = tried = 0
        last_error: ProviderError | None = None
        skipped = {"breaker": 0, "quota": 0, "bulkhead": 0, "budget": 0}

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

            reservation: Reservation | None = None
            if gate is not None and gate.is_paid(target):
                try:
                    reservation = await gate.reserve(target)
                except BudgetSkip:
                    self.bulkhead.release(name)
                    breaker.release_trial()
                    skipped["budget"] += 1
                    continue

            tried += 1
            lease = Lease(lambda n=name: self.bulkhead.release(n))
            retries = 0
            link_usage: dict | None = None  # what an invalid-output link cost, to charge it
            while True:
                if self.clock() >= deadline:
                    lease.close()
                    breaker.release_trial()
                    await _release(gate, reservation)
                    raise ChainFailed(last_error=last_error, tried=tried, timed_out=True, skipped=skipped)
                attempts += 1
                try:
                    attempt = await self._call(
                        target,
                        breaker,
                        lease,
                        tried + depth_offset,
                        deadline,
                        policy,
                        request,
                        stream,
                        validate,
                        repair_ok=attempts < MAX_ATTEMPTS,
                    )
                    attempts += attempt.extra_calls
                    attempt.gate, attempt.reservation = gate, reservation
                    if reservation is not None and attempt.completion is not None:
                        await gate.settle(reservation, attempt.completion.get("usage"))
                        attempt.reservation = None
                    return attempt
                except ProviderError as exc:
                    exc.provider = name
                    if exc.kind == "client":
                        lease.close()
                        breaker.release_trial()
                        await _release(gate, reservation)
                        raise
                    last_error = exc
                    if isinstance(exc, InvalidOutput):
                        attempts += exc.calls - 1
                        link_usage = exc.usage
                    else:
                        breaker.record_failure()
                    if exc.status in (429, 408):
                        self.bulkhead.feedback(name, True)  # AIMD: the provider is struggling, shrink
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
                    if reservation is not None:  # the call may have run: charge the input, not the output
                        await gate.settle(reservation, None)
                    raise
            lease.close()
            await _release(gate, reservation, link_usage)

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
        validate: OutputValidator | None = None,
        repair_ok: bool = True,
    ) -> Attempt:
        span = self.tracer.attempt(target.provider, target.model) if self.tracer else contextlib.nullcontext()
        try:
            with span:
                return await self._call_once(
                    target, breaker, lease, depth, deadline, policy, request, stream, validate, repair_ok
                )
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
        validate: OutputValidator | None = None,
        repair_ok: bool = True,
    ) -> Attempt:
        provider = self.providers[target.provider]
        started = self.clock()
        remaining = max(0.0, deadline - started)
        if not stream:
            # Without streaming there is no first token to time: the whole call is bounded by total_timeout.
            extra, problem = 0, None
            try:
                async with asyncio.timeout(remaining):
                    completion = await provider.complete(request, target.model)
                    problem = validate.check(completion) if validate is not None else None
                    if problem is not None and repair_ok:
                        extra = 1  # one repair retry on the same link, with the problem as context (R15)
                        second = await provider.complete(
                            repair_request(request, completion, problem), target.model
                        )
                        second["usage"] = merge_usage(completion.get("usage"), second.get("usage"))
                        completion, problem = second, validate.check(second)
            except TimeoutError as exc:
                raise ProviderError(
                    f"{target.provider}: no answer within {remaining:.1f}s", kind="transient", status=408
                ) from exc
            breaker.record_success()
            self._observe(target.provider, self.clock() - started)
            lease.close()
            if problem is not None:
                raise InvalidOutput(
                    f"{target.provider}: invalid structured output ({problem})",
                    usage=completion.get("usage"),
                    calls=1 + extra,
                )
            return Attempt(target, depth, breaker, lease, deadline, completion=completion, extra_calls=extra)

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
        ttft = self.clock() - started
        slow = ttft > SLOW_FRACTION * policy.ttft_timeout_s
        breaker.record_success(slow=slow)
        self._observe(target.provider, ttft, congested=slow)
        return Attempt(target, depth, breaker, lease, deadline, first=first, stream=gen)

    def _observe(self, provider: str, seconds: float, congested: bool = False) -> None:
        """A successful call: feed the latency EWMA (hedge delay, reordering) and the adaptive bulkhead."""
        if self.latency is not None:
            self.latency.record(provider, seconds)
        self.bulkhead.feedback(provider, congested)


async def _release(
    gate: BudgetGate | None, reservation: Reservation | None, usage: dict | None = None
) -> None:
    """The link produced nothing usable: give the hold back, charging only what `usage` says was spent."""
    if gate is not None and reservation is not None:
        if usage:
            await gate.settle(reservation, usage)
        else:
            await gate.settle(reservation, failed=True)


async def _aclose(gen: AsyncIterator[dict]) -> None:
    aclose = getattr(gen, "aclose", None)
    if aclose:
        await aclose()
