"""POST /v1/chat/completions: authenticate, admit, resolve the alias, run its chain, answer JSON or SSE."""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from llm_gateway.api.errors import GatewayError, request_id
from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.pricing.tokens import estimate_prompt_tokens
from llm_gateway.providers.base import ProviderError
from llm_gateway.resilience.admission import effective_priority
from llm_gateway.resilience.breaker import CircuitBreaker
from llm_gateway.resilience.executor import ChainFailed
from llm_gateway.resilience.singleflight import Flight, Shared, flight_key
from llm_gateway.routing.resolver import usable_chain
from llm_gateway.security.auth import require_client
from llm_gateway.state import GatewayState

log = logging.getLogger(__name__)
router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))}\n\n"


class _StreamAbort(Exception):
    """The stream cannot go on after its first token: a typed SSE error event, no fallback (ADR-4)."""

    def __init__(self, code: str, message: str, blame_provider: bool) -> None:
        super().__init__(message)
        self.code, self.blame_provider = code, blame_provider


class StreamCtx:
    """Everything a running stream must release or report, however it ends: finished, failed, or abandoned."""

    def __init__(
        self,
        *,
        include_usage: bool,
        rest: AsyncIterator[dict],
        cleanups: list[Callable[[], None]],
        deadline: float | None = None,
        stall_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        stop: asyncio.Future | None = None,
        breaker: CircuitBreaker | None = None,
        flight: Flight | None = None,
        label: str = "",
        depth: int = 1,
    ) -> None:
        self.include_usage, self.rest, self.cleanups = include_usage, rest, cleanups
        self.deadline, self.stall_s, self.clock, self.stop = deadline, stall_s, clock, stop
        self.breaker, self.flight, self.label, self.depth = breaker, flight, label, depth
        self.shared: Shared | None = None  # set when the stream ended cleanly
        self._closed = False

    async def close(self) -> None:
        """Idempotent. Runs from the generator's `finally` and from the response's background task, because a
        client that leaves before the first byte never starts the generator."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            aclose = getattr(self.rest, "aclose", None)  # stops the upstream call if the client disconnected
            if aclose:
                await aclose()
        for cleanup in self.cleanups:
            cleanup()
        if self.flight:
            self.flight.resolve(self.shared)  # None makes waiting followers run on their own
        # Budget settlement for aborted streams joins here in M5 (reserve → settle with the tokens produced).


async def _next_chunk(ctx: StreamCtx) -> dict | None:
    """The next chunk, within the stall timeout, the request's total timeout and the shutdown window."""
    if ctx.deadline is None:  # replay: no upstream to wait for
        return await anext(ctx.rest, None)
    loop = asyncio.get_running_loop()
    budget = min(ctx.stall_s, max(0.0, ctx.deadline - ctx.clock()))
    cb = None
    try:
        async with asyncio.timeout(budget) as scope:
            if ctx.stop is not None:

                def cb(_f: asyncio.Future) -> None:
                    with contextlib.suppress(RuntimeError):
                        scope.reschedule(loop.time())  # shutdown window closed: fire the timeout now

                if ctx.stop.done():
                    cb(ctx.stop)
                else:
                    ctx.stop.add_done_callback(cb)
            try:
                return await anext(ctx.rest, None)
            finally:
                if cb is not None:
                    ctx.stop.remove_done_callback(cb)
    except TimeoutError as exc:
        if ctx.stop is not None and ctx.stop.done():
            raise _StreamAbort(
                "server_shutting_down",
                "The gateway is shutting down; this response was cut short.",
                blame_provider=False,
            ) from exc
        if ctx.clock() >= ctx.deadline:
            raise _StreamAbort(
                "timeout", "The request exceeded its total timeout.", blame_provider=False
            ) from exc
        raise _StreamAbort(
            "stream_stalled", f"The provider sent nothing for {ctx.stall_s:g}s.", blame_provider=True
        ) from exc
    except ProviderError as exc:
        raise _StreamAbort("upstream_error", str(exc), blame_provider=True) from exc


async def _event_stream(first: dict, ctx: StreamCtx) -> AsyncIterator[str]:
    """Chunks as SSE. `usage` travels in the provider's last chunk; clients only get it when they ask."""
    usage, last, finish, text = None, first, None, []
    try:
        chunk: dict | None = first
        while chunk is not None:
            last = chunk
            usage = chunk.pop("usage", None) or usage
            for choice in chunk.get("choices") or []:
                text.append((choice.get("delta") or {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
            if chunk.get("choices"):
                yield _sse(chunk)
            chunk = await _next_chunk(ctx)
        meta = {
            "id": last.get("id", ""),
            "created": last.get("created", int(time.time())),
            "model": last.get("model", ""),
        }
        ctx.shared = Shared(
            {
                **meta,
                "object": "chat.completion",
                "usage": usage,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(text)},
                        "finish_reason": finish or "stop",
                    }
                ],
            },
            ctx.label,
            ctx.depth,
        )  # what a joined flight replays
        if ctx.include_usage and usage:
            yield _sse({**meta, "object": "chat.completion.chunk", "choices": [], "usage": usage})
        yield "data: [DONE]\n\n"
    except _StreamAbort as abort:
        log.warning("stream aborted (%s): %s", abort.code, abort)
        if abort.blame_provider and ctx.breaker is not None:
            ctx.breaker.record_failure()
        # No [DONE]: a stream that ends on an error event must not look like a clean completion.
        yield _sse({"error": {"message": str(abort), "type": "server_error", "code": abort.code}})
    finally:
        await ctx.close()


def _replay_chunks(completion: dict) -> AsyncIterator[dict]:
    """A finished completion as the chunks of a stream, for requests that joined a flight."""

    async def gen() -> AsyncIterator[dict]:
        meta = {
            "id": completion.get("id", ""),
            "object": "chat.completion.chunk",
            "created": completion.get("created", int(time.time())),
            "model": completion.get("model", ""),
        }
        choice = (completion.get("choices") or [{}])[0]
        yield {
            **meta,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield {
            **meta,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": (choice.get("message") or {}).get("content", "")},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            **meta,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
            "usage": completion.get("usage"),
        }

    return gen()


async def _replay(shared: Shared, body: ChatCompletionRequest, headers: dict, cleanups: list) -> Response:
    headers = {
        **headers,
        "X-Gateway-Provider": shared.provider,
        "X-Gateway-Fallback-Depth": str(shared.depth),
        "X-Gateway-Singleflight": "shared",
    }
    if not body.stream:
        return JSONResponse(shared.completion, headers=headers)
    chunks = _replay_chunks(shared.completion)
    ctx = StreamCtx(
        include_usage=bool(body.stream_options and body.stream_options.include_usage),
        rest=chunks,
        cleanups=list(cleanups),
    )
    cleanups.clear()
    first = await anext(chunks)
    return StreamingResponse(
        _event_stream(first, ctx),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, **headers},
        background=BackgroundTask(ctx.close),
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest, request: Request, client: Annotated[str, Depends(require_client)]
):
    gw: GatewayState = request.app.state.gateway
    rid = request_id(request)
    allowed = gw.clients.get(client).allowed_aliases
    policy = gw.routes.aliases.get(body.model)
    if policy is None:
        gw.stats.record("unknown", "-", 404)  # never label metrics with client-controlled text
        raise GatewayError(
            404,
            "model_not_found",
            f"The model `{body.model}` does not exist. Valid aliases: {', '.join(sorted(allowed))}.",
            param="model",
        )
    if body.model not in allowed:
        gw.stats.record(body.model, "-", 403)
        raise GatewayError(
            403, "alias_not_allowed", f"This API key may not use the alias `{body.model}`.", param="model"
        )

    if gw.lifecycle.draining:
        gw.stats.record(body.model, "-", 503)
        raise GatewayError(
            503,
            "server_shutting_down",
            "The gateway is shutting down; retry on another instance.",
            type_="server_error",
            headers={"X-Request-Id": rid, "Retry-After": "5"},
        )
    priority = effective_priority(policy.priority, request.headers.get("x-priority"))
    ticket = gw.admission.try_admit(priority)
    if ticket is None:
        gw.stats.record(body.model, "-", 503)
        raise GatewayError(
            503,
            "overloaded",
            f"The gateway is at capacity for `{priority}` requests; retry soon.",
            type_="server_error",
            headers={"X-Request-Id": rid, "Retry-After": "1" if priority == "interactive" else "5"},
        )
    cleanups: list[Callable[[], None]] = [
        ticket.release
    ]  # a stream takes these over and runs them at its end
    try:
        return await _serve(gw, body, client, policy, rid, cleanups)
    finally:
        for cleanup in cleanups:
            cleanup()


async def _serve(
    gw: GatewayState,
    body: ChatCompletionRequest,
    client: str,
    policy,
    rid: str,
    cleanups: list[Callable[[], None]],
) -> Response:
    chain = usable_chain(policy, gw.providers)
    max_tokens = policy.max_tokens if body.max_tokens is None else min(body.max_tokens, policy.max_tokens)
    upstream = body.model_copy(update={"max_tokens": max_tokens})
    base_headers = {"X-Request-Id": rid, "X-Alias": body.model}
    if body.max_tokens is not None and body.max_tokens > policy.max_tokens:
        base_headers["X-Max-Tokens-Clamped"] = str(policy.max_tokens)

    # Identical cacheable requests in flight share one provider call (singleflight, PLAN §4.7).
    flight: Flight | None = None
    if policy.cache if body.cache is None else body.cache:
        key = flight_key(
            client,
            body.model,
            [m.model_dump(exclude_none=True) for m in body.messages],
            body.response_format.model_dump(exclude_none=True) if body.response_format else None,
            max_tokens,
            body.temperature,
        )
        leader, joined = gw.singleflight.join(key)
        if leader:
            flight = joined
        else:
            shared = await joined.wait(policy.total_timeout_s)
            if shared is not None:
                gw.stats.record(body.model, shared.provider.split(":", 1)[0], 200)
                return await _replay(shared, body, base_headers, cleanups)
            # The leader failed: this request runs on its own, without a flight.

    try:
        attempt = await gw.executor.run(
            policy=policy,
            chain=chain,
            request=upstream,
            stream=body.stream,
            quota_tokens=estimate_prompt_tokens(body.messages) + max_tokens,
        )
    except ProviderError as exc:  # kind == "client": the request itself is invalid
        if flight:
            flight.resolve(None)
        gw.stats.record(body.model, exc.provider or "-", 400)
        raise GatewayError(400, "invalid_request", str(exc), headers={"X-Request-Id": rid}) from exc
    except ChainFailed as failed:
        if flight:
            flight.resolve(None)
        raise _chain_failure(gw, body.model, rid, failed) from failed
    except BaseException:
        if flight:
            flight.resolve(None)
        raise

    target = attempt.target
    label = f"{target.provider}:{target.model}"
    headers = {**base_headers, "X-Gateway-Provider": label, "X-Gateway-Fallback-Depth": str(attempt.depth)}
    gw.stats.record(body.model, target.provider, 200)
    if attempt.completion is not None:
        if flight:
            flight.resolve(Shared(attempt.completion, label, attempt.depth))
        return JSONResponse(attempt.completion, headers=headers)

    ctx = StreamCtx(
        include_usage=bool(body.stream_options and body.stream_options.include_usage),
        rest=attempt.stream,
        cleanups=[*cleanups, attempt.lease.close],
        deadline=attempt.deadline,
        stall_s=policy.stall_timeout_s,
        stop=gw.lifecycle.stop,
        breaker=attempt.breaker,
        flight=flight,
        label=label,
        depth=attempt.depth,
    )
    cleanups.clear()
    return StreamingResponse(
        _event_stream(attempt.first, ctx),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, **headers},
        background=BackgroundTask(ctx.close),
    )


def _chain_failure(gw: GatewayState, alias: str, rid: str, failed: ChainFailed) -> GatewayError:
    headers = {"X-Request-Id": rid}
    if failed.timed_out:
        gw.stats.record(alias, "-", 504)
        return GatewayError(
            504,
            "upstream_timeout",
            f"Alias `{alias}` did not answer within its total timeout.",
            type_="server_error",
            headers=headers,
        )
    gw.stats.record(alias, "-", 503)
    skipped = ", ".join(f"{n} by {why}" for why, n in failed.skipped.items() if n)
    registered = ", ".join(sorted(gw.providers)) or "none"
    if failed.last_error:
        detail = f" Last upstream error: {failed.last_error}."
    else:
        detail = f" No link could be tried (GW_ENV={gw.settings.env}, registered providers: {registered})."
    if skipped:
        detail += f" Skipped links: {skipped}."
    return GatewayError(
        503,
        "all_providers_unavailable",
        f"No provider could answer alias `{alias}`.{detail} "
        "Mock links exist only in dev and test; degraded links are skipped when the alias sets "
        "allow_degraded: false; providers with an open breaker, a full bulkhead or an exhausted "
        "quota are skipped.",
        type_="server_error",
        headers=headers,
    )
