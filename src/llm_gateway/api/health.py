from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm_gateway import __version__

router = APIRouter()


@router.get("/healthz")
def healthz(request: Request) -> dict:
    """Liveness plus provider health: an open breaker is visible here, and the gateway still answers."""
    gw = request.app.state.gateway
    return {"status": "ok", "version": __version__, "providers": gw.breakers.snapshot()}


@router.get("/readyz")
def readyz(request: Request):
    """Readiness: 503 once SIGTERM starts the drain, so the load balancer stops sending new requests."""
    if request.app.state.gateway.lifecycle.draining:
        return JSONResponse({"status": "draining"}, status_code=503)
    return {"status": "ready"}
