"""Global admission control: load shedding by priority (PLAN §4.7)."""

from typing import Literal

Priority = Literal["interactive", "batch"]


class Ticket:
    """One request in flight. `release` may be called any number of times."""

    def __init__(self, admission: "Admission") -> None:
        self._admission, self._released = admission, False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._admission.in_flight -= 1


class Admission:
    def __init__(self, limit: int = 64, batch_ratio: float = 0.75) -> None:
        self.limit = limit
        self.batch_limit = int(limit * batch_ratio)  # 48 of 64
        self.in_flight = 0

    def try_admit(self, priority: Priority) -> Ticket | None:
        """`batch` is turned away first, from 75% of the limit; `interactive` only at 100%."""
        cap = self.limit if priority == "interactive" else self.batch_limit
        if self.in_flight >= cap:
            return None
        self.in_flight += 1
        return Ticket(self)


def effective_priority(alias_priority: Priority, header: str | None) -> Priority:
    """A client may lower its priority (`X-Priority: batch`) but never raise it."""
    if alias_priority == "batch" or (header or "").strip().lower() == "batch":
        return "batch"
    return "interactive"
