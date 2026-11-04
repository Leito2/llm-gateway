from fastapi import APIRouter, Request

from llm_gateway.api.schemas import ModelCard, ModelList

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
def list_models(request: Request) -> ModelList:
    routes = request.app.state.gateway.routes
    return ModelList(data=[ModelCard(id=alias) for alias in sorted(routes.aliases)])
