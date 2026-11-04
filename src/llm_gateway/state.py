"""Dependencies shared by the request handlers (stored in `app.state.gateway`)."""

from dataclasses import dataclass, field

import httpx

from llm_gateway.budget.service import BudgetService
from llm_gateway.cache.embedder import Embedder
from llm_gateway.cache.service import CacheService
from llm_gateway.guardrails.promptguard import PromptGuard
from llm_gateway.pricing.prices import PriceBook
from llm_gateway.providers.base import Provider
from llm_gateway.quotas.limiter import QuotaLimiter
from llm_gateway.resilience.admission import Admission
from llm_gateway.resilience.breaker import BreakerRegistry
from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.resilience.executor import Executor
from llm_gateway.resilience.lifecycle import Lifecycle
from llm_gateway.resilience.singleflight import Singleflight
from llm_gateway.routing.config import RoutesConfig
from llm_gateway.routing.latency import LatencyTracker
from llm_gateway.security.clients import ClientRegistry
from llm_gateway.security.ratelimit import RateLimiter
from llm_gateway.settings import Settings
from llm_gateway.telemetry.langfuse import LangfuseExporter
from llm_gateway.telemetry.metrics import Metrics
from llm_gateway.telemetry.stats import Stats
from llm_gateway.telemetry.tracing import Tracer


@dataclass
class GatewayState:
    settings: Settings
    routes: RoutesConfig
    providers: dict[str, Provider]
    clients: ClientRegistry
    quotas: QuotaLimiter
    stats: Stats
    http: httpx.AsyncClient
    breakers: BreakerRegistry
    bulkhead: Bulkhead
    admission: Admission
    singleflight: Singleflight
    lifecycle: Lifecycle
    executor: Executor
    embedder: Embedder | None = None
    cache: CacheService | None = None
    prices: PriceBook | None = None
    budget: BudgetService | None = None
    metrics: Metrics | None = None
    ratelimit: RateLimiter | None = None
    tracer: Tracer | None = None
    latency: LatencyTracker = field(default_factory=LatencyTracker)
    langfuse: LangfuseExporter | None = None
    prompt_guard: PromptGuard | None = None
