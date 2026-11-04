"""G12: what hedging does to tail latency. The primary answers in ~20 ms, but 5% of its calls hang for 1 s;
a healthy secondary answers in ~20 ms. Same seeded load with `hedge` off and on.

    uv run python -m bench.hedging [--requests 1500] [--concurrency 16] [--delay 0.08]

In-process (httpx ASGITransport), so numbers are about the gateway's decision logic and real sleeps, not
about a network. Pass: p99 with hedging at least 3x lower than without, at the cost of extra calls on the
secondary (reported as `hedge_rate`)."""

import argparse
import asyncio
import random
import statistics
import sys
import time

import httpx

from bench.common import save

sys.path.insert(0, "tests")

FAST_S, SLOW_S, SLOW_RATE = 0.02, 1.0, 0.05


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))] * 1000, 1)


async def run_once(hedge: bool, requests: int, concurrency: int, delay: float, seed: int) -> dict:
    from helpers import FULL_KEY, make_app

    from llm_gateway.providers.mock import MockProvider
    from llm_gateway.quotas.limiter import QuotaLimiter

    rng = random.Random(seed)  # same sequence of slow calls in both runs

    class Tailed(MockProvider):
        async def complete(self, request, model):
            await asyncio.sleep(SLOW_S if rng.random() < SLOW_RATE else FAST_S)
            return await super().complete(request, model)

    class Steady(MockProvider):
        async def complete(self, request, model):
            await asyncio.sleep(FAST_S)
            return await super().complete(request, model)

    app = make_app()
    gw = app.state.gateway
    gw.providers["groq"], gw.providers["mock"] = Tailed(), Steady()
    gw.executor.quotas = QuotaLimiter({})  # the free-tier rpm of config/providers.yaml is not under test
    policy = gw.routes.aliases["fast"]
    policy.chain[-1].degraded = False
    policy.hedge, policy.hedge_delay_s = hedge, delay
    policy.total_timeout_s = 10.0
    events: list[str] = []
    gw.executor.on_hedge = events.append
    latencies: list[float] = []
    ok = 0
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bench",
        headers={"Authorization": f"Bearer {FULL_KEY}"},
        timeout=30,
    ) as client:

        async def one(i: int) -> None:
            nonlocal ok
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(
                    "/v1/chat/completions",
                    json={"model": "fast", "messages": [{"role": "user", "content": f"q{i}"}]},
                )
                latencies.append(time.perf_counter() - t0)
                ok += r.status_code == 200

        await asyncio.gather(*(one(i) for i in range(requests)))
    return {
        "hedge": hedge,
        "success": ok,
        "mean_ms": round(statistics.fmean(latencies) * 1000, 1),
        "p50_ms": pct(latencies, 0.50),
        "p95_ms": pct(latencies, 0.95),
        "p99_ms": pct(latencies, 0.99),
        "hedges_launched": events.count("launched"),
        "hedge_rate": round(events.count("launched") / requests, 3),
        "hedges_won": events.count("won"),
    }


async def main(requests: int, concurrency: int, delay: float, seed: int) -> dict:
    off = await run_once(False, requests, concurrency, delay, seed)
    on = await run_once(True, requests, concurrency, delay, seed)
    ratio = round(off["p99_ms"] / on["p99_ms"], 2) if on["p99_ms"] else None
    return {
        "scenario": {
            "fast_ms": FAST_S * 1000,
            "slow_ms": SLOW_S * 1000,
            "slow_rate": SLOW_RATE,
            "hedge_delay_s": delay,
        },
        "without_hedging": off,
        "with_hedging": on,
        "p99_improvement_x": ratio,
        "pass": bool(ratio and ratio >= 3 and on["success"] == requests),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--delay", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    result = asyncio.run(main(args.requests, args.concurrency, args.delay, args.seed))
    print(save("g12_hedging", result).read_text(encoding="utf-8"))
    sys.exit(0 if result["pass"] else 1)
