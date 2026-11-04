from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
_UI = Path(__file__).resolve().parent.parent / "ui"
# The page keeps the user's API key in sessionStorage: it may only talk to its own origin.
_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'"
)
# The guide also loads Google Fonts; everything else stays locked down.
_GUIDE_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; connect-src 'self'; base-uri 'none'; form-action 'none'"
)


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_UI / "index.html", media_type="text/html", headers={"Content-Security-Policy": _CSP})


@router.get("/guide", include_in_schema=False)
def guide() -> FileResponse:
    """The documentation page (phases, features, run instructions) with a chat that talks to this gateway."""
    return FileResponse(
        _UI / "guide.html", media_type="text/html", headers={"Content-Security-Policy": _GUIDE_CSP}
    )
