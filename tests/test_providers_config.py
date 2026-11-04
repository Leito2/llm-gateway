from pathlib import Path

import httpx
import pytest

from llm_gateway.providers.config import load_providers
from llm_gateway.providers.mock import MockProvider
from llm_gateway.providers.openai_compat import OpenAICompatProvider
from llm_gateway.providers.registry import build_providers
from llm_gateway.routing.config import load_routes
from llm_gateway.settings import Settings

CONFIG = load_providers(Path("config/providers.yaml"))


def build(**settings) -> dict:
    s = Settings(_env_file=None, **{"groq_api_key": None, "google_api_key": None, **settings})
    return build_providers(s, CONFIG, httpx.AsyncClient())


def test_shipped_config_is_consistent_with_the_routes():
    CONFIG.check_routes(load_routes(Path("config/routes.yaml")))


def test_undeclared_provider_in_a_chain_stops_startup():
    routes = load_routes(Path("config/routes.yaml"))
    routes.aliases["fast"].chain[0].provider = "typo"
    with pytest.raises(ValueError, match="typo"):
        CONFIG.check_routes(routes)


def test_without_keys_only_ollama_and_mock_exist_in_dev():
    assert set(build(env="dev")) == {"mock", "ollama"}


def test_mock_is_missing_in_prod():
    assert set(build(env="prod")) == {"ollama"}


def test_a_key_registers_its_provider():
    providers = build(env="prod", groq_api_key="k")
    assert set(providers) == {"ollama", "groq"} and isinstance(providers["groq"], OpenAICompatProvider)
    assert not isinstance(providers["groq"], MockProvider)


def test_paid_providers_need_both_a_key_and_a_run_budget():
    assert "anthropic" not in build(env="dev", anthropic_api_key="k", run_budget_usd=0.0)
    assert "anthropic" not in build(env="dev", anthropic_api_key=None, run_budget_usd=3.0)
    assert "anthropic" in build(env="dev", anthropic_api_key="k", run_budget_usd=3.0)


def test_quotas_come_from_the_yaml():
    assert CONFIG.providers["groq"].quotas.rpm == 30 and CONFIG.providers["google"].quotas.rpd == 1000
