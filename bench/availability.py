"""G10: availability with a sick primary. 10,000 requests; the primary provider fails 20% of calls with a 503,
answers 10% with a 429 and sits on 10% for longer than the TTFT timeout. A healthy secondary stands behind it.

    uv run python -m bench.availability [--requests 10000] [--concurrency 32]

In-process (httpx ASGITransport, no sockets), so it is deterministic and fast enough for the nightly run.
Pass: success rate >= 99.9% (PLAN section 10)."""

import argparse
import asyncio
import random
import sys
import time

import httpx

from bench.common import save

sys.path.insert(0, "tests")


async def main(requests: int, concurrency: int, seed: int) -> dict:
    from helpers import FULL_KEY, make_app

    from llm_gateway.providers.base import ProviderError
    from llm_gateway.providers.mock import MockProvider

    rng = random.Random(seed)

    class SickPrimary(MockProvider):
        calls = 0

        async def complete(self, request, model):
            SickPrimary.calls += 1
            roll = rng.random()
            if roll < 0.20:
                raise ProviderError("injected 503", kind="transient", status=503)
            if roll < 0.30:
                raise ProviderError("injected 429", kind="transient", status=429, retry_after=1)
            if roll < 0.40:
                await asyncio.sleep(1.0)  # longer than the 0.2 s timeout below: the call is abandoned
            return await super().complete(request, model)

    app = make_app()
    gw = app.state.gateway
    primary = SickPrimary()
    gw.providers["groq"] = primary
    gw.routes.aliases["fast"].total_timeout_s = 0.2
    gw.executor.sleep = lambda _s: asyncio.sleep(0)  # no real backoff: we measure outcomes, not waiting
    ok, codes = 0, {}
    sem = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://bench", headers={"Authorization": f"Bearer {FULL_KEY}"}
    ) as client:

        async def one(i: int) -> None:
            nonlocal ok
            async with sem:
                r = await client.post(
                    "/v1/chat/completions",
                    json={"model": "fast", "messages": [{"role": "user", "content": f"q{i}"}]},
                )
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
            ok += r.status_code == 200

        await asyncio.gather(*(one(i) for i in range(requests)))
    elapsed = time.perf_counter() - started
    return {
        "requests": requests,
        "success": ok,
        "success_rate": round(ok / requests, 5),
        "status_codes": codes,
        "primary_calls": primary.calls,
        "breaker_primary": gw.breakers.get("groq").snapshot(),
        "seconds": round(elapsed, 1),
        "pass": ok / requests >= 0.999,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=10_000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    result = asyncio.run(main(args.requests, args.concurrency, args.seed))
    print(save("g10_availability", result).read_text(encoding="utf-8"))
    sys.exit(0 if result["pass"] else 1)
