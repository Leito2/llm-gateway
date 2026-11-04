"""Dependencies shared by the request handlers (stored in `app.state.gateway`)."""
from dataclasses import dataclass

import httpx

from llm_gateway.providers.base import Provider
from llm_gateway.quotas.limiter import QuotaLimiter
from llm_gateway.resilience.admission import Admission
from llm_gateway.resilience.breaker import BreakerRegistry
from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.resilience.executor import Executor
from llm_gateway.resilience.lifecycle import Lifecycle
from llm_gateway.resilience.singleflight import Singleflight
from llm_gateway.routing.config import RoutesConfig
from llm_gateway.security.clients import ClientRegistry
from llm_gateway.settings import Settings
from llm_gateway.telemetry.stats import Stats


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
