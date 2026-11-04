"""G1: what the gateway adds to a request. Open-loop load against a zero-latency upstream, direct vs through
the gateway, in two flavours: no cache, and a cache MISS with the semantic layer on (embedding + KNN).

    uv run python -m bench.overhead [--rate 100] [--semantic-rate 20] [--seconds 15] [--embedder e5-small]

Honest caveats: the load generator is Python (httpx) on the same machine as the gateway, the upstream and the
embedder, so absolute numbers include its overhead and CPU contention; the DIFFERENCE between direct and
gateway is the figure to read. The semantic miss runs at a lower rate on purpose: one ONNX embedding per
request is CPU work, and past the embedder's capacity the lookup times out and the request is served as a
plain miss (that is the designed degradation, see `semantic lookup skipped` in the gateway log)."""

import argparse
import asyncio
import uuid

from bench.common import KEY, body, gateway_stack, open_loop, save

AUTH = {"Authorization": f"Bearer {KEY}"}


def unique(alias: str):
    return lambda i: body(alias, f"unique question {uuid.uuid4()} number {i}")


def diff(a: dict, b: dict) -> dict:
    return {p: round(a[p] - b[p], 2) for p in ("p50", "p95", "p99")}


async def main(rate: float, semantic_rate: float, seconds: float, embedder: str) -> dict:
    out: dict = {"seconds": seconds, "embedder": embedder}
    with gateway_stack(embedder=embedder) as (gw, up):
        chat_up, chat_gw = f"{up}/v1/chat/completions", f"{gw}/v1/chat/completions"
        await open_loop(chat_up, unique("m"), 50, 2)  # warm-up
        await open_loop(chat_gw, unique("bench"), 50, 2, AUTH)
        await open_loop(chat_gw, unique("bench_sem"), 20, 2, AUTH)

        direct = await open_loop(chat_up, unique("m"), rate, seconds)
        plain = await open_loop(chat_gw, unique("bench"), rate, seconds, AUTH)
        out["no_cache"] = {
            "rate_rps": rate,
            "direct": direct,
            "gateway": plain,
            "overhead_ms": diff(plain, direct),
        }

        direct = await open_loop(chat_up, unique("m"), semantic_rate, seconds)
        plain = await open_loop(chat_gw, unique("bench"), semantic_rate, seconds, AUTH)
        miss = await open_loop(chat_gw, unique("bench_sem"), semantic_rate, seconds, AUTH)
        out["semantic_miss"] = {
            "rate_rps": semantic_rate,
            "direct": direct,
            "gateway_no_cache": plain,
            "gateway_semantic_miss": miss,
            "overhead_ms": diff(miss, direct),
            "semantic_cost_ms": diff(miss, plain),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=100)
    ap.add_argument("--semantic-rate", type=float, default=20)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--embedder", default="e5-small")
    args = ap.parse_args()
    result = asyncio.run(main(args.rate, args.semantic_rate, args.seconds, args.embedder))
    print(save("g1_overhead", result).read_text(encoding="utf-8"))
