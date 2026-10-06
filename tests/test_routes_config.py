from pathlib import Path

from llm_gateway.routing.config import load_routes

ROUTES = load_routes(Path("config/routes.yaml"))


def test_paid_provider_only_in_paid_alias():
    for alias, policy in ROUTES.aliases.items():
        providers = {t.provider for t in policy.chain}
        if alias != "smart_paid":
            assert "anthropic" not in providers, f"{alias} must not route to a paid provider"


def test_every_chain_ends_offline():
    """Inherited from the Go gateway: the gateway must keep answering without internet."""
    for alias, policy in ROUTES.aliases.items():
        assert policy.chain[-1].provider in {"ollama", "mock"}, f"{alias} must end in a local provider"


def test_judges_never_use_semantic_cache():
    for alias in ("judge", "judge_golden"):
        assert ROUTES.aliases[alias].semantic_cache is False
