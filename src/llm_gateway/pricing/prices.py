"""Prices and costs in integer micro-dollars (ADR-0004).

`prices.yaml` states USD per 1M tokens. 1 USD = 1,000,000 micro-dollars, so 0.10 USD/Mtok is exactly
0.10 micro-dollars per token. Costs are rounded UP: recorded spend is never below real spend."""

import math
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Price(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None  # defaults to `input`
    cache_write: float | None = None  # defaults to `input`
    reference_input: float = 0.0  # what a free tier would cost if it were paid (savings reports only)
    reference_output: float = 0.0

    @property
    def paid(self) -> bool:
        return self.input > 0 or self.output > 0


class Usage(BaseModel):
    """Token counts, OpenAI convention: `prompt_tokens` INCLUDES cached and cache-creation tokens."""

    prompt: int = Field(default=0, ge=0)
    completion: int = Field(default=0, ge=0)
    cached: int = Field(default=0, ge=0)  # served from the provider's prompt cache
    cache_creation: int = Field(default=0, ge=0)  # written to it

    @classmethod
    def from_completion(cls, usage: dict | None) -> "Usage | None":
        if not usage:
            return None
        details = usage.get("prompt_tokens_details") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = min(int(details.get("cached_tokens") or 0), prompt)
        creation = min(int(details.get("cache_creation_tokens") or 0), prompt - cached)
        return cls(
            prompt=prompt,
            completion=int(usage.get("completion_tokens") or 0),
            cached=cached,
            cache_creation=creation,
        )


def _micro(total: Decimal) -> int:
    return math.ceil(total)


def _d(x: float | None, default: float) -> Decimal:
    return Decimal(str(default if x is None else x))


def cost_micro(price: Price, usage: Usage) -> int:
    """What the provider charges. Free tiers: 0."""
    plain = usage.prompt - usage.cached - usage.cache_creation
    total = (
        plain * _d(price.input, 0)
        + usage.cached * _d(price.cache_read, price.input)
        + usage.cache_creation * _d(price.cache_write, price.input)
        + usage.completion * _d(price.output, 0)
    )
    return _micro(total)


def reference_micro(price: Price, usage: Usage) -> int:
    """Cost at paid rates: the real cost for paid providers, the reference price for free ones."""
    if price.paid:
        return cost_micro(price, usage)
    return _micro(
        usage.prompt * _d(price.reference_input, 0) + usage.completion * _d(price.reference_output, 0)
    )


def max_cost_micro(price: Price, input_tokens: int, max_tokens: int, prompt_cache: bool = False) -> int:
    """Worst case for one request: all input at the dearest input rate, all `max_tokens` produced (thinking is
    billed as output inside `max_tokens`, so it is covered)."""
    rate = _d(price.input, 0)
    if prompt_cache:
        rate = max(rate, _d(price.cache_write, price.input))
    return _micro(input_tokens * rate + max_tokens * _d(price.output, 0))


class PriceBook:
    def __init__(self, prices: dict[str, dict[str, Price]]) -> None:
        self._prices = prices

    def price(self, provider: str, model: str) -> Price:
        models = self._prices.get(provider, {})
        return models.get(model) or models.get("default") or Price()


def load_prices(path: Path) -> PriceBook:
    if not path.exists():
        return PriceBook({})
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PriceBook(
        {p: {m: Price.model_validate(v) for m, v in models.items()} for p, models in raw.items()}
    )
