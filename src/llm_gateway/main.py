"""FastAPI application factory."""
import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api import admin, chat, health, models, ui
from llm_gateway.api.errors import install_error_handlers
from llm_gateway.providers.config import load_providers
from llm_gateway.providers.registry import HTTP_TIMEOUT, build_providers
from llm_gateway.quotas.limiter import QuotaLimiter
from llm_gateway.resilience.admission import Admission
from llm_gateway.resilience.breaker import BreakerRegistry
from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.resilience.executor import Executor
from llm_gateway.resilience.lifecycle import Lifecycle
from llm_gateway.resilience.singleflight import Singleflight
from llm_gateway.routing.config import load_routes
from llm_gateway.security.clients import load_clients
from llm_gateway.settings import Settings
from llm_gateway.state import GatewayState
from llm_gateway.telemetry.stats import Stats


@asynccontextmanager
async def _lifespan(app: FastAPI):
    gw = app.state.gateway
    gw.lifecycle.attach(asyncio.get_running_loop())
    yield
    await gw.http.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    routes = load_routes(settings.routes_file)
    provider_config = load_providers(settings.providers_file)
    provider_config.check_routes(routes)
    clients = load_clients(settings.clients_file)
    clients.check_routes(routes)
    http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    providers = build_providers(settings, provider_config, http)
    quotas = QuotaLimiter({name: spec.quotas for name, spec in provider_config.providers.items()})
    breakers = BreakerRegistry()
    bulkhead = Bulkhead({name: spec.concurrency for name, spec in provider_config.providers.items()
                         if spec.concurrency})

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=_lifespan)
    app.state.gateway = GatewayState(
        settings=settings,
        routes=routes,
        providers=providers,
        clients=clients,
        quotas=quotas,
        stats=Stats(),
        http=http,
        breakers=breakers,
        bulkhead=bulkhead,
        admission=Admission(settings.max_in_flight),
        singleflight=Singleflight(),
        lifecycle=Lifecycle(settings.drain_timeout_s),
        executor=Executor(providers, breakers, bulkhead, quotas),
    )
    install_error_handlers(app)
    for module in (health, models, chat, admin, ui):
        app.include_router(module.router)
    return app


app = create_app()
