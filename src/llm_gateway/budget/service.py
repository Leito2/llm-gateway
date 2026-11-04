"""Budget policy on top of a store: who may spend how much, and what a request can cost (PLAN §5)."""

import logging
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

from llm_gateway.budget.store import BudgetStore, BudgetUnavailable, Caps, Reservation, Scopes
from llm_gateway.pricing.prices import PriceBook, Usage, cost_micro, max_cost_micro, reference_micro
from llm_gateway.pricing.tokens import estimate_tokens
from llm_gateway.routing.config import AliasPolicy, Target

log = logging.getLogger(__name__)

RESERVATION_GRACE_S = 30.0  # a reservation outlives the alias total_timeout by this much (ADR-0004)


class BudgetSkip(Exception):
    """This paid link cannot be used now: the reservation did not fit, or the store is down."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def micro(usd: float | None) -> int | None:
    return None if usd is None else round(usd * 1_000_000)


class BudgetService:
    def __init__(
        self,
        store: BudgetStore,
        prices: PriceBook,
        paid_providers: set[str],
        *,
        run_cap_usd: float | None,
        global_day_cap_usd: float | None = None,
        today: Callable[[], str] = lambda: datetime.now(UTC).date().isoformat(),
        on_cost: Callable[[str, str, int, int], None] | None = None,
    ) -> None:
        self.store, self.prices, self.paid = store, prices, paid_providers
        self.run_cap, self.global_cap = micro(run_cap_usd), micro(global_day_cap_usd)
        self._today = today
        self.on_cost = on_cost  # (client, provider, cost, reference cost) after every settle
        self.counters: Counter[str] = Counter()

    def scopes(self, client: str) -> Scopes:
        day = self._today()
        return Scopes("budget:run", f"budget:day:{day}:client:{client}", f"budget:day:{day}:global")

    def gate(
        self,
        client: str,
        client_day_cap_usd: float | None,
        policy: AliasPolicy,
        prompt_tokens: int,
        max_tokens: int,
        prompt_cache: bool,
    ) -> "BudgetGate":
        caps = Caps(self.run_cap, micro(client_day_cap_usd), self.global_cap)
        ttl = policy.total_timeout_s + RESERVATION_GRACE_S
        return BudgetGate(self, client, caps, ttl, prompt_tokens, max_tokens, prompt_cache)

    async def snapshot(self, client: str, client_day_cap_usd: float | None) -> dict:
        caps = Caps(self.run_cap, micro(client_day_cap_usd), self.global_cap)
        try:
            usage = await self.store.snapshot(self.scopes(client))
        except BudgetUnavailable:
            return {"available": False}
        cap_of = {"run": caps.run, "client_day": caps.client_day, "global_day": caps.global_day}
        return {
            "available": True,
            "unit": "micro-dollars (1 USD = 1,000,000)",
            "scopes": {
                n: {
                    **v,
                    "cap": cap_of[n],
                    "remaining": None if cap_of[n] is None else cap_of[n] - v["spent"] - v["reserved"],
                }
                for n, v in usage.items()
            },
            "counters": dict(self.counters),
        }


class BudgetGate:
    """One request's view of the budget. Free providers pass through; paid ones must reserve first."""

    def __init__(
        self,
        svc: BudgetService,
        client: str,
        caps: Caps,
        ttl_s: float,
        prompt_tokens: int,
        max_tokens: int,
        prompt_cache: bool,
    ) -> None:
        self.svc, self.client, self.caps, self.ttl_s = svc, client, caps, ttl_s
        self.prompt_tokens, self.max_tokens, self.prompt_cache = prompt_tokens, max_tokens, prompt_cache
        self.skipped: list[str] = []

    def is_paid(self, target: Target) -> bool:
        return target.provider in self.svc.paid

    async def reserve(self, target: Target) -> Reservation:
        """Hold the worst-case cost of a call to `target`. Raises BudgetSkip."""
        price = self.svc.prices.price(target.provider, target.model)
        amount = max_cost_micro(price, self.prompt_tokens, self.max_tokens, self.prompt_cache)
        scopes, rid = self.svc.scopes(self.client), uuid.uuid4().hex
        try:
            ok = await self.svc.store.reserve(rid, amount, self.ttl_s, scopes, self.caps)
        except BudgetUnavailable as exc:
            self.svc.counters["unavailable"] += 1
            self.skipped.append(target.provider)
            raise BudgetSkip("unavailable") from exc
        if not ok:
            self.svc.counters["rejected"] += 1
            self.skipped.append(target.provider)
            raise BudgetSkip("cap")
        self.svc.counters["reserved"] += 1
        return Reservation(rid, amount, scopes, target.provider, target.model, self.prompt_tokens)

    async def settle(
        self, res: Reservation, usage: dict | None = None, *, emitted_tokens: int = 0, failed: bool = False
    ) -> tuple[int, int]:
        """Replace the hold with the real cost. Returns (cost, reference cost) in micro-dollars.

        `failed`: the provider produced nothing, so nothing is charged. No `usage` (cancelled stream, provider
        that does not report): the whole input plus the output seen, rounded up."""
        price = self.svc.prices.price(res.provider, res.model)
        parsed = Usage.from_completion(usage)
        if failed:
            real = ref = 0
        else:
            if parsed is None:
                parsed = Usage(prompt=res.max_input, completion=emitted_tokens)
                self.svc.counters["usage_estimated"] += 1
            real, ref = cost_micro(price, parsed), reference_micro(price, parsed)
        try:
            late = await self.svc.store.settle(res.id, real, res.scopes)
        except BudgetUnavailable:
            # The hold stays until its deadline frees it; the spend is lost to the ledger, so shout about it.
            self.svc.counters["settle_failed"] += 1
            log.error("budget settle failed for %s: %d micro-dollars unrecorded", res.provider, real)
            return real, ref
        if late:
            self.svc.counters["late_settle"] += 1
        if self.svc.on_cost:
            self.svc.on_cost(self.client, res.provider, real, ref)
        return real, ref


def estimate_emitted(text_chars: int) -> int:
    return estimate_tokens("x" * text_chars)
