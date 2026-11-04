"""Request size limit (size/shape guardrail, PLAN §8.4): oversized bodies get 413 before they are parsed."""

import json

from starlette.types import ASGIApp, Receive, Scope, Send


class BodyLimitMiddleware:
    """Pure ASGI, so a rejected body is never read. Judges `Content-Length` (normal clients send it)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length", b"0")
            if length.isdigit() and int(length) > self.max_bytes:
                body = json.dumps(
                    {
                        "error": {
                            "message": f"The request body is larger than {self.max_bytes} bytes.",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "request_too_large",
                        }
                    }
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
