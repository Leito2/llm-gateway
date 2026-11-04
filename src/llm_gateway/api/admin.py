from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/stats")
def stats(request: Request) -> dict:
    """Request counters. M8 puts this behind the admin key; until then it exposes nothing but counts."""
    gw = request.app.state.gateway
    return {**gw.stats.snapshot(), "breakers": gw.breakers.snapshot(), "bulkheads": gw.bulkhead.snapshot(),
            "in_flight": {"now": gw.admission.in_flight, "limit": gw.admission.limit,
                          "batch_limit": gw.admission.batch_limit},
            "singleflight_in_flight": gw.singleflight.in_flight, "draining": gw.lifecycle.draining}
