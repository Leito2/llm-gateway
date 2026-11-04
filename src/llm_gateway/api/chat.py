"""POST /v1/chat/completions: authenticate, resolve the alias, walk its chain, answer JSON or SSE."""
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.api.errors import GatewayError, request_id
from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.pricing.tokens import estimate_prompt_tokens
from llm_gateway.providers.base import ProviderError
from llm_gateway.routing.resolver import usable_chain
from llm_gateway.security.auth import require_client

log = logging.getLogger(__name__)
router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_BLOCK_AFTER_429_S = 30.0     # when the provider sends a 429 without Retry-After


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
            meta = {"id": last.get("id", ""), "object": "chat.completion.chunk",
                    "created": last.get("created", int(time.time())), "model": last.get("model", "")}
            yield _sse({**meta, "choices": [], "usage": usage})
        yield "data: [DONE]\n\n"
    finally:
        aclose = getattr(rest, "aclose", None)   # closes the upstream stream if the client disconnects
        if aclose:
            await aclose()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request,
                           client: Annotated[str, Depends(require_client)]):
    gw = request.app.state.gateway
    rid = request_id(request)
    allowed = gw.clients.get(client).allowed_aliases
    policy = gw.routes.aliases.get(body.model)
    if policy is None:
        gw.stats.record("unknown", "-", 404)     # never label metrics with client-controlled text
        raise GatewayError(404, "model_not_found", f"The model `{body.model}` does not exist. "
                           f"Valid aliases: {', '.join(sorted(allowed))}.", param="model")
    if body.model not in allowed:
        gw.stats.record(body.model, "-", 403)
        raise GatewayError(403, "alias_not_allowed", f"This API key may not use the alias `{body.model}`.",
                           param="model")

    chain = usable_chain(policy, gw.providers)
    max_tokens = policy.max_tokens if body.max_tokens is None else min(body.max_tokens, policy.max_tokens)
    clamped = body.max_tokens is not None and body.max_tokens > policy.max_tokens
    upstream = body.model_copy(update={"max_tokens": max_tokens})
    quota_tokens = estimate_prompt_tokens(body.messages) + max_tokens
    base_headers = {"X-Request-Id": rid, "X-Alias": body.model}
    if clamped:
        base_headers["X-Max-Tokens-Clamped"] = str(policy.max_tokens)

    # Sequential fallback. M3 replaces this loop with timeouts per phase, retries, breakers and attempt caps.
    tried, last_error = 0, None
    for target in chain:
        if not gw.quotas.try_acquire(target.provider, quota_tokens):
            continue                              # quota exhausted: move on before the provider says 429
        tried += 1
        provider = gw.providers[target.provider]
        headers = {**base_headers, "X-Gateway-Provider": f"{target.provider}:{target.model}",
                   "X-Gateway-Fallback-Depth": str(tried)}
        try:
            if not body.stream:
                result = await provider.complete(upstream, target.model)
                gw.stats.record(body.model, target.provider, 200)
                return JSONResponse(result, headers=headers)
            # Take the first chunk before answering: until a token is out, a failure can still fall back
            # (ADR-4).
            stream = provider.stream(upstream, target.model)
            first = await anext(stream)
        except ProviderError as exc:
            if exc.kind == "client":
                gw.stats.record(body.model, target.provider, 400)
                raise GatewayError(400, "invalid_request", str(exc), headers={"X-Request-Id": rid}) from exc
            if exc.status == 429:
                gw.quotas.block(target.provider, exc.retry_after or _BLOCK_AFTER_429_S)
            log.warning("provider %s failed (%s): %s", target.provider, exc.kind, exc)
            last_error = exc
            continue
        gw.stats.record(body.model, target.provider, 200)
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return StreamingResponse(_event_stream(first, stream, include_usage), media_type="text/event-stream",
                                 headers={**_SSE_HEADERS, **headers})

    gw.stats.record(body.model, "-", 503)
    registered = ", ".join(sorted(gw.providers)) or "none"
    detail = (f" Last upstream error: {last_error}." if last_error
              else f" No link could be tried (GW_ENV={gw.settings.env}, registered providers: {registered}).")
    raise GatewayError(503, "all_providers_unavailable",
                       f"No provider could answer alias `{body.model}`.{detail} Mock links exist only in dev "
                       "and test; degraded links are skipped when the alias sets allow_degraded: false; "
                       "providers over their quota are skipped.",
                       type_="server_error", headers={"X-Request-Id": rid})
