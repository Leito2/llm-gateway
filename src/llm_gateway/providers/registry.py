"""Build the provider registry. Providers that are not registered are skipped by the router."""
from llm_gateway.providers.base import Provider
from llm_gateway.providers.mock import MockProvider
from llm_gateway.settings import Settings


def build_providers(settings: Settings) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    if settings.env in ("dev", "test"):
        providers["mock"] = MockProvider()
    return providers
