from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from llm_gateway.api.errors import GatewayError
from llm_gateway.cache.keys import parse_tags
from llm_gateway.security.admin import client_or_admin, require_admin

router = APIRouter()


@router.get("/stats", dependencies=[Depends(require_admin)])
def stats(request: Request) -> dict:
    """Request counters. M8 puts this behind the admin key; until then it exposes nothing but counts."""
    gw = request.app.state.gateway
    return {
        **gw.stats.snapshot(),
        "breakers": gw.breakers.snapshot(),
        "bulkheads": gw.bulkhead.snapshot(),
        "in_flight": {
            "now": gw.admission.in_flight,
            "limit": gw.admission.limit,
            "batch_limit": gw.admission.batch_limit,
        },
        "singleflight_in_flight": gw.singleflight.in_flight,
        "draining": gw.lifecycle.draining,
        "cache": gw.cache.snapshot() if gw.cache else None,
    }


class InvalidateRequest(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=50)
    client: str | None = None  # admin only: limit to one client (default: every client)


@router.post("/admin/cache/invalidate")
async def invalidate_cache(
    body: InvalidateRequest, request: Request, caller: Annotated[str | None, Depends(client_or_admin)]
) -> dict:
    """Drop every cached answer that carries any of the tags (R09). A client drops only its own entries; the
    admin key drops every client's (or one client's, with `client`)."""
    client = caller if caller is not None else body.client
    gw = request.app.state.gateway
    if gw.cache is None:
        raise GatewayError(
            503, "cache_disabled", "The cache is off (GW_CACHE_BACKEND=off).", type_="server_error"
        )
    tags = set(parse_tags(",".join(body.tags)))
    if not tags:
        raise GatewayError(
            400, "invalid_request", "No valid tag. Tags use letters, digits and `:_.-/`.", param="tags"
        )
    return {"removed": await gw.cache.invalidate(tags, client=client), "tags": sorted(tags)}


@router.get("/admin/budget")
async def budget(
    request: Request, caller: Annotated[str | None, Depends(client_or_admin)], client: str | None = None
) -> dict:
    """Spend, holds and remaining room for the run, a client's day and the global day, plus cost and savings
    counters. A client sees itself; the admin key may name any client with `?client=`."""
    gw = request.app.state.gateway
    client = caller or client or "-"
    cfg = gw.clients.get(client)
    return {
        "client": client,
        **await gw.budget.snapshot(client, cfg.budget_usd_per_day if cfg else None),
        "cost_micro_usd": gw.stats.snapshot()["cost_micro_usd"],
        "cache_saved_micro_usd": gw.stats.snapshot()["cache_saved_micro_usd"],
    }
