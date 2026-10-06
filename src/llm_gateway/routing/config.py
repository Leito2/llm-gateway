"""Load and validate config/routes.yaml (aliases → ordered fallback chains)."""
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Target(BaseModel):
    provider: str                       # "mock" | "ollama" | "anthropic"
    model: str


class AliasPolicy(BaseModel):
    chain: list[Target] = Field(min_length=1)
    max_tokens: int = 512
    ttft_timeout_s: float = 10.0
    stall_timeout_s: float = 10.0
    total_timeout_s: float = 120.0
    cache: bool = False
    semantic_cache: bool = False


class RoutesConfig(BaseModel):
    aliases: dict[str, AliasPolicy]


def load_routes(path: Path) -> RoutesConfig:
    return RoutesConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
