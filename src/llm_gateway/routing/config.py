"""Load and validate config/routes.yaml (aliases → ordered fallback chains)."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Target(BaseModel):
    provider: str                       # key in config/providers.yaml
    model: str


class AliasPolicy(BaseModel):
    chain: list[Target] = Field(min_length=1)
    max_tokens: int = 512
    ttft_timeout_s: float = 10.0
    stall_timeout_s: float = 10.0
    total_timeout_s: float = 120.0
    cache: bool = False
    semantic_cache: bool = True         # mandatory by default (PLAN §6); opt-out per alias
    hedge: bool = False
    priority: Literal["interactive", "batch"] = "interactive"


class RoutesConfig(BaseModel):
    aliases: dict[str, AliasPolicy]


def load_routes(path: Path) -> RoutesConfig:
    return RoutesConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
