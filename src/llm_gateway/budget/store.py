"""Atomic budget accounting: reserve the worst case, settle the real cost (PLAN §5.2, ADR-0004).

Three scopes are checked together: the whole run, the client's day and the global day (00:00 UTC start).
If any cap does not fit, nothing is reserved. Amounts are integer micro-dollars. Reservations expire
logically (a deadline, not a Redis TTL): `reserve` first frees the expired ones."""

from dataclasses import dataclass
from typing import Protocol


class BudgetUnavailable(Exception):
    """The store cannot be reached. Paid providers fail closed (ADR-7); free ones keep working."""


@dataclass(frozen=True)
class Caps:
    """Micro-dollar caps; None means no cap on that scope."""

    run: int | None
    client_day: int | None
    global_day: int | None


@dataclass(frozen=True)
class Scopes:
    run: str
    client_day: str
    global_day: str

    def as_list(self) -> list[str]:
        return [self.run, self.client_day, self.global_day]


@dataclass(frozen=True)
class Reservation:
    id: str
    amount: int
    scopes: Scopes
    provider: str
    model: str
    max_input: int  # tokens, for a conservative settle when `usage` is missing


class BudgetStore(Protocol):
    async def reserve(self, rid: str, amount: int, ttl_s: float, scopes: Scopes, caps: Caps) -> bool:
        """True and the amount is held; False and nothing is held. Raises BudgetUnavailable."""
        ...

    async def settle(self, rid: str, real: int, scopes: Scopes) -> bool:
        """Release the hold and add `real` to spent. True when the reservation had already expired (late)."""
        ...

    async def snapshot(self, scopes: Scopes) -> dict: ...
