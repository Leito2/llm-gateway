"""FastAPI application factory."""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api import admin, chat, health, models, ui
from llm_gateway.api.errors import install_error_handlers
from llm_gateway.providers.config import load_providers
from llm_gateway.providers.registry import HTTP_TIMEOUT, build_providers
from llm_gateway.quotas.limiter import QuotaLimiter
from llm_gateway.routing.config import load_routes
from llm_gateway.security.clients import load_clients
from llm_gateway.settings import Settings
from llm_gateway.state import GatewayState
from llm_gateway.telemetry.stats import Stats


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    await app.state.gateway.http.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    routes = load_routes(settings.routes_file)
    provider_config = load_providers(settings.providers_file)
    provider_config.check_routes(routes)
    clients = load_clients(settings.clients_file)
    clients.check_routes(routes)
    http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=_lifespan)
    app.state.gateway = GatewayState(
        settings=settings,
        routes=routes,
        providers=build_providers(settings, provider_config, http),
        clients=clients,
        quotas=QuotaLimiter({name: spec.quotas for name, spec in provider_config.providers.items()}),
        stats=Stats(),
        http=http,
    )
    install_error_handlers(app)
    for module in (health, models, chat, admin, ui):
        app.include_router(module.router)
    return app


app = create_app()
