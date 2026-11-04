"""Load and validate config/routes.yaml (aliases → ordered fallback chains)."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Target(BaseModel):
    provider: str                       # key in config/providers.yaml
    model: str
    degraded: bool = False              # low-quality last resort: never cached (PLAN §3.2)


class AliasPolicy(BaseModel):
    chain: list[Target] = Field(min_length=1)
    max_tokens: int = 512
    ttft_timeout_s: float = 10.0
    stall_timeout_s: float = 10.0
    total_timeout_s: float = 120.0
    cache: bool = False                 # exact cache only (L1)
    semantic_cache: bool = True         # exact + semantic (L1, then L2): on by default (PLAN §6), opt-out
    cache_ttl_s: int = Field(default=86400, gt=0)
    semantic_scope: Literal["single", "last_turn"] = "single"   # last_turn: every user question stands alone
    hedge: bool = False
    allow_degraded: bool = True         # False: never fall back to a `degraded` link
    priority: Literal["interactive", "batch"] = "interactive"


class RoutesConfig(BaseModel):
    aliases: dict[str, AliasPolicy]


def load_routes(path: Path) -> RoutesConfig:
    return RoutesConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
