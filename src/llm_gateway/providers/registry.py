"""Build the provider registry. A provider that is not registered is skipped by the router."""
import logging

import httpx

from llm_gateway.providers.base import Provider
from llm_gateway.providers.config import ProvidersConfig, ProviderSpec
from llm_gateway.providers.mock import MockProvider
from llm_gateway.providers.openai_compat import OpenAICompatProvider
from llm_gateway.settings import Settings

log = logging.getLogger(__name__)

# connect: a dead provider must fail fast. read: longest silence between bytes (per-phase timeouts: M3).
HTTP_TIMEOUT = httpx.Timeout(connect=2.0, read=60.0, write=10.0, pool=5.0)


def _setting(settings: Settings, env_name: str | None) -> str | None:
    """`GW_GROQ_API_KEY` -> settings.groq_api_key"""
    return getattr(settings, env_name.removeprefix("GW_").lower(), None) if env_name else None


def build_providers(settings: Settings, config: ProvidersConfig,
                    http: httpx.AsyncClient) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    for name, spec in config.providers.items():
        provider = _build(name, spec, settings, http)
        if provider is not None:
            providers[name] = provider
    return providers


def _build(name: str, spec: ProviderSpec, settings: Settings, http: httpx.AsyncClient) -> Provider | None:
    if spec.kind == "mock":
        return MockProvider() if settings.env in ("dev", "test") else None
    if spec.paid or spec.kind == "anthropic":
        log.info("provider %s not registered: paid providers arrive with the budget (M5/M7)", name)
        return None
    api_key = _setting(settings, spec.api_key_env)
    if spec.api_key_env and not api_key:
        log.info("provider %s not registered: %s is not set", name, spec.api_key_env)
        return None
    base_url = spec.base_url or _setting(settings, spec.base_url_env)
    if not base_url:
        log.warning("provider %s not registered: no base_url", name)
        return None
    return OpenAICompatProvider(name, base_url, api_key, http)
