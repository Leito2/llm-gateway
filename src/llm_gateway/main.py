"""FastAPI application factory."""
from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api import admin, chat, health, models, ui
from llm_gateway.api.errors import install_error_handlers
from llm_gateway.providers.registry import build_providers
from llm_gateway.routing.config import load_routes
from llm_gateway.settings import Settings
from llm_gateway.state import GatewayState
from llm_gateway.telemetry.stats import Stats


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="llm-gateway", version=__version__)
    app.state.gateway = GatewayState(
        settings=settings,
        routes=load_routes(settings.routes_file),
        providers=build_providers(settings),
        stats=Stats(),
    )
    install_error_handlers(app)
    for module in (health, models, chat, admin, ui):
        app.include_router(module.router)
    return app


app = create_app()
