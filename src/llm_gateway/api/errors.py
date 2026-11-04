"""OpenAI-format errors: {"error": {"message", "type", "param", "code"}}."""
import re
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def request_id(request: Request) -> str:
    """The client's `X-Request-ID` when it is safe to echo, otherwise a new id."""
    rid = getattr(request.state, "request_id", None)
    if rid is None:
        raw = request.headers.get("x-request-id", "")
        rid = raw if _REQUEST_ID.fullmatch(raw) else uuid.uuid4().hex
        request.state.request_id = rid
    return rid


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, *, type_: str = "invalid_request_error",
                 param: str | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.type, self.param, self.headers = type_, param, headers or {}


def _error_response(status: int, code: str, message: str, *, type_: str, param: str | None,
                    headers: dict[str, str]) -> JSONResponse:
    body = {"error": {"message": message, "type": type_, "param": param, "code": code}}
    return JSONResponse(body, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        headers = {"X-Request-Id": request_id(request), **exc.headers}
        return _error_response(exc.status, exc.code, exc.message, type_=exc.type, param=exc.param,
                               headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p != "body")
        message = f"{where}: {first['msg']}" if where else first["msg"]
        return _error_response(400, "invalid_request", message, type_="invalid_request_error",
                               param=where or None, headers={"X-Request-Id": request_id(request)})
