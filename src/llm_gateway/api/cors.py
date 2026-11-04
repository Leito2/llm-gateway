"""Opt-in CORS (PLAN 8.3 keeps it off by default: the bundled UI is served from the same origin).

`GW_CORS_ORIGINS=https://my.app,http://localhost:5173` (or `*`) lets a browser app on another origin call
the gateway, for example a hosted chat page talking to a gateway on your own machine. Chrome also asks
private-network permission for public pages calling localhost (`allow_private_network`).
Keys still travel only in the `Authorization` header."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

EXPOSED = [
    "X-Request-Id", "X-Alias", "X-Gateway-Provider", "X-Gateway-Fallback-Depth", "X-Gateway-Singleflight",
    "X-Cache-Layer", "X-Cache-Similarity", "X-Max-Tokens-Clamped", "X-Params-Dropped", "X-Effort",
    "X-Structured-Repaired", "X-Budget-Skipped", "Retry-After",
]  # fmt: skip


def install_cors(app: FastAPI, origins: str) -> None:
    allowed = [o.strip() for o in origins.split(",") if o.strip()]
    if not allowed:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Priority",
            "X-Cache-Scope",
            "X-Cache-Tags",
            "X-Prompt-Cache",
            "X-Request-ID",
            "traceparent",
        ],
        expose_headers=EXPOSED,
        allow_credentials=False,
        allow_private_network=True,  # Chrome asks before a public page may call localhost
    )
