"""GET /metrics: Prometheus text format, behind the admin key (PLAN §8.1)."""

from fastapi import APIRouter, Depends, Request, Response

from llm_gateway.security.admin import require_admin

router = APIRouter()


@router.get("/metrics", dependencies=[Depends(require_admin)])
async def metrics(request: Request) -> Response:
    gw = request.app.state.gateway
    return Response(await gw.metrics.render(gw), media_type="text/plain; version=0.0.4; charset=utf-8")
