"""FastAPI dependency: `Authorization: Bearer <key>` -> client name, or 401."""
from fastapi import Request

from llm_gateway.api.errors import GatewayError

_CHALLENGE = {"WWW-Authenticate": "Bearer"}


async def require_client(request: Request) -> str:
    gw = request.app.state.gateway
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    key = key.strip()
    if scheme.lower() != "bearer" or not key:
        gw.stats.record("-", "-", 401)
        raise GatewayError(401, "missing_api_key", "Send your key as `Authorization: Bearer <key>`.",
                           headers=_CHALLENGE)
    name = gw.clients.authenticate(key)
    if name is None:
        gw.stats.record("-", "-", 401)
        raise GatewayError(401, "invalid_api_key", "Incorrect API key provided.", headers=_CHALLENGE)
    request.state.client = name
    return name
