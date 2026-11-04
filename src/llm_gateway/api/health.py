from fastapi import APIRouter

from llm_gateway import __version__

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__}
