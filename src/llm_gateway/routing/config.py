"""Load and validate config/routes.yaml (aliases → ordered fallback chains)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Target(BaseModel):
    provider: str  # key in config/providers.yaml
    model: str
    degraded: bool = False  # low-quality last resort: never cached (PLAN §3.2)


class AliasPolicy(BaseModel):
    chain: list[Target] = Field(min_length=1)
    max_tokens: int = 512
    max_messages: int = Field(default=200, gt=0)  # size/shape guardrail (PLAN §8.4)
    max_images: int = Field(default=8, ge=0)
    ttft_timeout_s: float = 10.0
    stall_timeout_s: float = 10.0
    total_timeout_s: float = 120.0
    cache: bool = False  # exact cache only (L1)
    semantic_cache: bool = True  # exact + semantic (L1, then L2): on by default (PLAN §6), opt-out
    cache_ttl_s: int = Field(default=86400, gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)  # longer requests get a 400 before any call
    effort: Literal["low", "medium", "high"] | None = None  # Anthropic: upper bound for reasoning effort
    prompt_cache: bool = False  # mark the system prompt for the provider's prompt cache (Anthropic)
    semantic_scope: Literal["single", "last_turn"] = "single"  # last_turn: every user question stands alone
    hedge: bool = False  # race the next link when the first is slower than `hedge_delay_s` (stretch)
    hedge_delay_s: float | None = Field(default=None, gt=0)  # default: 3x the provider's EWMA TTFT
    reorder: Literal["none", "ewma"] = "none"  # ewma: fastest non-degraded link first (stretch)
    vision: bool = False  # accepts image parts; other aliases answer 400 `images_not_supported`
    allow_degraded: bool = True  # False: never fall back to a `degraded` link
    priority: Literal["interactive", "batch"] = "interactive"


class RoutesConfig(BaseModel):
    aliases: dict[str, AliasPolicy]


def load_routes(path: Path) -> RoutesConfig:
    return RoutesConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
