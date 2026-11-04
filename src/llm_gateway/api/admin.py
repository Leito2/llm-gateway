from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/stats")
def stats(request: Request) -> dict:
    """Request counters. M8 puts this behind the admin key; until then it exposes nothing but counts."""
    return request.app.state.gateway.stats.snapshot()
