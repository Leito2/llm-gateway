from pathlib import Path

from llm_gateway.routing.config import load_routes


def test_paid_provider_only_in_paid_alias():
    routes = load_routes(Path("config/routes.yaml"))
    for alias, policy in routes.aliases.items():
        providers = {t.provider for t in policy.chain}
        if alias != "smart_paid":
            assert "anthropic" not in providers, f"{alias} must not route to a paid provider"
