from typing import Annotated

from fastapi import APIRouter, Depends, Request

from llm_gateway.api.schemas import ModelCard, ModelList
from llm_gateway.security.auth import require_client

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
def list_models(request: Request, client: Annotated[str, Depends(require_client)]) -> ModelList:
    """The aliases this client may use."""
    gw = request.app.state.gateway
    allowed = set(gw.clients.get(client).allowed_aliases)
    return ModelList(data=[ModelCard(id=alias) for alias in sorted(gw.routes.aliases) if alias in allowed])
