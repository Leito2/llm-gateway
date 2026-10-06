"""FastAPI application factory. M0: health and model listing; chat completions arrive in M1."""
from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api.schemas import ModelCard, ModelList
from llm_gateway.routing.config import load_routes
from llm_gateway.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    routes = load_routes(settings.routes_file)
    app = FastAPI(title="llm-gateway", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models", response_model=ModelList)
    def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=alias) for alias in sorted(routes.aliases)])

    return app


app = create_app()
