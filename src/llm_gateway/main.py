"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api import admin, chat, embeddings, health, models, ui
from llm_gateway.api import metrics as metrics_api
from llm_gateway.api.cors import install_cors
from llm_gateway.api.errors import install_error_handlers
from llm_gateway.api.limits import BodyLimitMiddleware
from llm_gateway.budget.memory import MemoryBudget
from llm_gateway.budget.service import BudgetService
from llm_gateway.cache.config import load_cache_config
from llm_gateway.cache.embedder import build_embedder
from llm_gateway.cache.memory import MemoryBackend
from llm_gateway.cache.service import CacheService
from llm_gateway.guardrails.promptguard import load_prompt_guard
from llm_gateway.pricing.prices import load_prices
from llm_gateway.providers.config import load_providers
from llm_gateway.providers.registry import HTTP_TIMEOUT, build_providers
from llm_gateway.quotas.limiter import QuotaLimiter
from llm_gateway.reload import watch
from llm_gateway.resilience.admission import Admission
from llm_gateway.resilience.breaker import BreakerRegistry
from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.resilience.executor import Executor
from llm_gateway.resilience.lifecycle import Lifecycle
from llm_gateway.resilience.singleflight import Singleflight
from llm_gateway.routing.config import load_routes
from llm_gateway.routing.latency import LatencyTracker
from llm_gateway.security.clients import load_clients
from llm_gateway.security.ratelimit import RateLimiter
from llm_gateway.settings import Settings
from llm_gateway.state import GatewayState
from llm_gateway.telemetry.metrics import Metrics
from llm_gateway.telemetry.privacy import check_prompt_logging
from llm_gateway.telemetry.record import configure_access_log
from llm_gateway.telemetry.stats import Stats
from llm_gateway.telemetry.tracing import build_tracer, export_targets


@asynccontextmanager
async def _lifespan(app: FastAPI):
    gw = app.state.gateway
    gw.lifecycle.attach(asyncio.get_running_loop())
    watcher = None
    if gw.settings.reload_interval_s > 0:
        watcher = asyncio.create_task(watch(gw, gw.settings.reload_interval_s))
    yield
    if watcher is not None:
        watcher.cancel()
    if gw.cache is not None:
        await gw.cache.flush()
    await gw.http.aclose()


def create_app(settings: Settings | None = None, tracer_provider=None) -> FastAPI:
    settings = settings or Settings()
    check_prompt_logging(settings)
    configure_access_log(settings.access_log)
    routes = load_routes(settings.routes_file)
    provider_config = load_providers(settings.providers_file)
    provider_config.check_routes(routes)
    clients = load_clients(settings.clients_file)
    clients.check_routes(routes)
    http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    providers = build_providers(settings, provider_config, http)
    quotas = QuotaLimiter({name: spec.quotas for name, spec in provider_config.providers.items()})
    breakers = BreakerRegistry()
    prices = load_prices(settings.prices_file)
    metrics = Metrics()
    stats = Stats(metrics)
    if settings.budget_backend == "redis":
        import redis.asyncio as aioredis

        from llm_gateway.budget.redis_store import RedisBudget

        store = RedisBudget(aioredis.from_url(settings.redis_url))
    else:
        store = MemoryBudget()
    budget = BudgetService(
        store,
        prices,
        {n for n, spec in provider_config.providers.items() if spec.paid},
        run_cap_usd=settings.run_budget_usd,
        global_day_cap_usd=settings.global_day_budget_usd,
        on_cost=stats.record_cost,
    )
    cache_config = load_cache_config(settings.cache_file)
    embedder = None if settings.embedder == "off" else build_embedder(settings.embedder, settings.models_dir)
    cache = None
    if settings.cache_backend != "off":
        cache = CacheService(_build_cache_backend(settings, cache_config), cache_config, embedder)
    tracer = build_tracer(export_targets(settings), tracer_provider)
    bulkhead = Bulkhead(
        {name: spec.concurrency for name, spec in provider_config.providers.items() if spec.concurrency},
        adaptive={
            name: spec.min_concurrency
            for name, spec in provider_config.providers.items()
            if spec.concurrency and spec.adaptive
        },
    )
    latency = LatencyTracker()
    executor = Executor(
        providers, breakers, bulkhead, quotas, tracer=tracer, latency=latency, on_hedge=metrics.hedges_hook
    )

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=_lifespan)
    app.state.gateway = GatewayState(
        settings=settings,
        routes=routes,
        providers=providers,
        clients=clients,
        quotas=quotas,
        stats=stats,
        http=http,
        breakers=breakers,
        bulkhead=bulkhead,
        admission=Admission(settings.max_in_flight),
        singleflight=Singleflight(),
        lifecycle=Lifecycle(settings.drain_timeout_s),
        executor=executor,
        embedder=embedder,
        cache=cache,
        prices=prices,
        budget=budget,
        metrics=metrics,
        ratelimit=_build_ratelimit(settings),
        tracer=tracer,
        latency=latency,
        prompt_guard=load_prompt_guard(settings.models_dir / "prompt-guard")
        if settings.guardrails_mode != "off"
        else None,
    )
    install_error_handlers(app)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    install_cors(app, settings.cors_origins)
    for module in (health, models, chat, embeddings, admin, metrics_api, ui):
        app.include_router(module.router)
    return app


def _build_cache_backend(settings: Settings, cache_config):
    if settings.cache_backend == "redis":
        import redis.asyncio as aioredis

        from llm_gateway.cache.redis_backend import RedisBackend

        return RedisBackend(aioredis.from_url(settings.redis_url))
    return MemoryBackend(cache_config.max_entries)


def _build_ratelimit(settings: Settings) -> RateLimiter:
    if settings.rate_limit_backend == "redis":
        import redis.asyncio as aioredis

        return RateLimiter(redis=aioredis.from_url(settings.redis_url))
    return RateLimiter()


app = create_app()
