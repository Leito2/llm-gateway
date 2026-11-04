from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
_INDEX = Path(__file__).resolve().parent.parent / "ui" / "index.html"
# The page keeps the user's API key in sessionStorage: it may only talk to its own origin.
_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'")


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_INDEX, media_type="text/html", headers={"Content-Security-Policy": _CSP})
