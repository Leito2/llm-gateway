"""Backoff with full jitter (PLAN §4.2): wait = rand(0, min(cap, base * 2^attempt))."""

import random

BASE_S = 0.1
CAP_S = 0.5


def backoff_s(
    attempt: int, *, base: float = BASE_S, cap: float = CAP_S, rng: random.Random | None = None
) -> float:
    """`attempt` counts retries from 0. Full jitter keeps simultaneous failures from retrying in lockstep."""
    return (rng or random).uniform(0.0, min(cap, base * 2**attempt))
