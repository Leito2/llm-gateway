"""POST /v1/chat/completions: resolve the alias, call the first usable provider, answer JSON or SSE."""
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.api.errors import GatewayError, request_id
from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.routing.resolver import usable_chain

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def _event_stream(first: dict, rest: AsyncIterator[dict], include_usage: bool) -> AsyncIterator[str]:
    """Chunks as SSE. `usage` travels in the provider's last chunk; clients only get it when they ask."""
    usage, last = None, first
    try:
        chunk: dict | None = first
        while chunk is not None:
            last = chunk
            usage = chunk.pop("usage", None) or usage
            if chunk.get("choices"):
                yield _sse(chunk)
            chunk = await anext(rest, None)
        if include_usage and usage:
            meta = {k: last[k] for k in ("id", "object", "created", "model")}
            yield _sse({**meta, "choices": [], "usage": usage})
        yield "data: [DONE]\n\n"
    finally:
        aclose = getattr(rest, "aclose", None)   # closes the upstream stream if the client disconnects
        if aclose:
            await aclose()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    gw = request.app.state.gateway
    rid = request_id(request)
    policy = gw.routes.aliases.get(body.model)
    if policy is None:
        gw.stats.record("unknown", "-", 404)     # never label metrics with client-controlled text
        valid = ", ".join(sorted(gw.routes.aliases))
        raise GatewayError(404, "model_not_found",
                           f"The model `{body.model}` does not exist. Valid aliases: {valid}.", param="model")

    chain = usable_chain(policy, gw.providers)
    if not chain:
        gw.stats.record(body.model, "-", 503)
        registered = ", ".join(sorted(gw.providers)) or "none"
        raise GatewayError(503, "all_providers_unavailable",
                           f"No usable provider for alias `{body.model}`. GW_ENV={gw.settings.env}, "
                           f"registered providers: {registered}. Mock links exist only in dev and test; "
                           "degraded links are skipped when the alias sets allow_degraded: false.",
                           type_="server_error")
    target = chain[0]    # M3 walks the whole chain with retries, breakers and fallback
    provider = gw.providers[target.provider]

    max_tokens = policy.max_tokens if body.max_tokens is None else min(body.max_tokens, policy.max_tokens)
    clamped = body.max_tokens is not None and body.max_tokens > policy.max_tokens
    upstream = body.model_copy(update={"max_tokens": max_tokens})
    headers = {
        "X-Request-Id": rid,
        "X-Alias": body.model,
        "X-Gateway-Provider": f"{target.provider}:{target.model}",
        "X-Gateway-Fallback-Depth": "1",
    }
    if clamped:
        headers["X-Max-Tokens-Clamped"] = str(policy.max_tokens)

    if not body.stream:
        result = await provider.complete(upstream, target.model)
        gw.stats.record(body.model, target.provider, 200)
        return JSONResponse(result, headers=headers)

    # Take the first chunk before answering: a failure up to here can still become a plain HTTP error
    # (and, from M3, a fallback). After the first token nothing may be retried (ADR-4).
    stream = provider.stream(upstream, target.model)
    first = await anext(stream)
    gw.stats.record(body.model, target.provider, 200)
    include_usage = bool(body.stream_options and body.stream_options.include_usage)
    return StreamingResponse(_event_stream(first, stream, include_usage), media_type="text/event-stream",
                             headers={**_SSE_HEADERS, **headers})
