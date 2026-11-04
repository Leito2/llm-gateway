"""Load and validate config/providers.yaml."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from llm_gateway.quotas.limiter import QuotaSpec
from llm_gateway.routing.config import RoutesConfig


class ProviderSpec(BaseModel):
    kind: Literal["mock", "ollama", "openai_compat", "anthropic"]
    base_url: str | None = None
    base_url_env: str | None = None  # GW_* variable that holds the URL (Settings field of the same name)
    api_key_env: str | None = None  # GW_* variable with the key; if it is empty the provider is unused
    quotas: QuotaSpec = Field(default_factory=QuotaSpec)
    concurrency: int | None = Field(default=None, gt=0)
    paid: bool = False
    adaptive: bool = False  # AIMD: `concurrency` becomes the maximum and moves down under congestion
    min_concurrency: int = Field(default=1, gt=0)


class ProvidersConfig(BaseModel):
    providers: dict[str, ProviderSpec]

    def check_routes(self, routes: RoutesConfig) -> None:
        """A chain that names an undeclared provider must stop the start-up, not fail under traffic."""
        for alias, policy in routes.aliases.items():
            for target in policy.chain:
                if target.provider not in self.providers:
                    raise ValueError(
                        f"alias {alias!r} uses provider {target.provider!r}, "
                        "which is not declared in providers.yaml"
                    )


def load_providers(path: Path) -> ProvidersConfig:
    return ProvidersConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
