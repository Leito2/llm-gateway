"""Admin key (PLAN §8.1/8.3): a key separate from client keys, stored only as a SHA-256 hash.

`GW_ADMIN_KEY_SHA256` set: `/stats`, `/metrics` and `/admin/*` need it, in every environment.
Unset: those routes are open in dev and test (convenience) and closed in prod (fail-closed)."""

import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, Request

from llm_gateway.api.errors import GatewayError
from llm_gateway.security.auth import require_client

_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def _bearer(request: Request) -> str:
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    return key.strip() if scheme.lower() == "bearer" else ""


def is_admin(request: Request) -> bool:
    """True when the request carries the admin key."""
    expected = request.app.state.gateway.settings.admin_key_sha256
    key = _bearer(request)
    if not expected or not key:
        return False
    return hmac.compare_digest(hashlib.sha256(key.encode()).hexdigest(), expected.strip().lower())


async def require_admin(request: Request) -> None:
    gw = request.app.state.gateway
    if gw.settings.admin_key_sha256:
        if not _bearer(request):
            raise GatewayError(
                401,
                "missing_admin_key",
                "Send the admin key as `Authorization: Bearer <key>`.",
                headers=_CHALLENGE,
            )
        if not is_admin(request):
            raise GatewayError(401, "invalid_admin_key", "Incorrect admin key.", headers=_CHALLENGE)
        return
    if gw.settings.env == "prod":
        raise GatewayError(
            403, "admin_disabled", "Set GW_ADMIN_KEY_SHA256 to use this endpoint.", type_="server_error"
        )


async def client_or_admin(request: Request) -> str | None:
    """None for an admin (who names the client explicitly), otherwise the authenticated client."""
    if is_admin(request):
        return None
    return await require_client(request)


AdminOnly = Annotated[None, Depends(require_admin)]
