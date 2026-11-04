"""FastAPI application factory."""
import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api import admin, chat, embeddings, health, models, ui
from llm_gateway.api.errors import install_error_handlers
from llm_gateway.cache.config import load_cache_config
from llm_gateway.cache.embedder import build_embedder
from llm_gateway.cache.memory import MemoryBackend
from llm_gateway.cache.service import CacheService
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
    if gw.cache is not None:
        await gw.cache.flush()
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
    cache_config = load_cache_config(settings.cache_file)
    embedder = None if settings.embedder == "off" else build_embedder(settings.embedder)
    cache = (
        None
        if settings.cache_backend == "off"
        else CacheService(MemoryBackend(cache_config.max_entries), cache_config, embedder)
    )
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
        embedder=embedder,
        cache=cache,
    )
    install_error_handlers(app)
    for module in (health, models, chat, embeddings, admin, ui):
        app.include_router(module.router)
    return app


app = create_app()
