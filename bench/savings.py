"""G11: how much of the bill the cache removes on Zipf-shaped FAQ traffic.

    uv run python -m bench.savings [--requests 5000] [--questions 300] [--zipf 1.1]

In-process, exact cache only: the semantic layer is still uncalibrated (it observes, never serves), so this is
the floor the semantic layer would add to. "Cost" is the REFERENCE price of the answering model (free tiers
cost $0 but have a reference price in config/prices.yaml), exactly what `gw_cost_saved_usd_total` reports.
Two traffic shapes: verbatim repeats (1 wording per question) and paraphrase noise (3 wordings per question,
chosen at random), where an exact cache cannot recognise the other wordings.
The question popularity is a synthetic Zipf, not real P3/P2 traffic; the PLAN's 30-40 % claim is a claim about
real traffic and is NOT confirmed by this number."""

import argparse
import asyncio
import random
import sys

import httpx

from bench.common import save

sys.path.insert(0, "tests")

PROMPT_TOKENS, COMPLETION_TOKENS = 60, 180


def zipf_weights(n: int, s: float) -> list[float]:
    return [1 / (rank**s) for rank in range(1, n + 1)]


async def replay(requests: int, questions: int, zipf: float, wordings: int, seed: int) -> dict:
    from helpers import FULL_KEY, make_app

    from llm_gateway.providers.mock import MockProvider
    from llm_gateway.routing.config import Target

    class Priced(MockProvider):
        calls = 0

        async def complete(self, request, model):
            Priced.calls += 1
            out = await super().complete(request, model)
            out["usage"] = {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            }
            return out

    Priced.calls = 0
    app = make_app(cache_backend="memory", embedder="off")
    gw = app.state.gateway
    gw.providers["groq"] = Priced()
    policy = gw.routes.aliases["fast"]
    policy.chain = [Target(provider="groq", model="llama-3.3-70b-versatile")]
    gw.executor.quotas = type(gw.quotas)({})  # free-tier rpm is not under test

    rng = random.Random(seed)
    weights = zipf_weights(questions, zipf)
    hits = 0
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bench",
        headers={"Authorization": f"Bearer {FULL_KEY}"},
        timeout=30,
    ) as client:
        for _ in range(requests):
            q = rng.choices(range(questions), weights)[0]
            text = f"faq {q} wording {rng.randrange(wordings)}: how do I do thing number {q}?"
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "fast", "messages": [{"role": "user", "content": text}]},
            )
            hits += r.headers.get("x-cache-layer") is not None
            await gw.cache.flush()  # the write-back is async: wait for it so a repeat can hit
        await gw.cache.flush()
        stats = (await client.get("/stats")).json()
    saved = sum(stats["cache_saved_micro_usd"].values())
    spent = sum(row["reference"] for row in stats["cost_micro_usd"])
    total = saved + spent
    return {
        "wordings_per_question": wordings,
        "requests": requests,
        "provider_calls": Priced.calls,
        "hit_rate": round(hits / requests, 4),
        "reference_cost_without_cache_usd": round(total / 1e6, 4),
        "reference_cost_with_cache_usd": round(spent / 1e6, 4),
        "cost_saved_pct": round(100 * saved / total, 1) if total else None,
    }


async def main(requests: int, questions: int, zipf: float, seed: int) -> dict:
    return {
        "traffic": {
            "zipf_s": zipf,
            "questions": questions,
            "tokens_per_call": [PROMPT_TOKENS, COMPLETION_TOKENS],
        },
        "verbatim_repeats": await replay(requests, questions, zipf, 1, seed),
        "paraphrase_noise": await replay(requests, questions, zipf, 3, seed),
        "long_tail_paraphrase_noise": await replay(
            requests, questions * 10, zipf, 3, seed
        ),  # 10x more questions
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    print(
        save("g11_savings", asyncio.run(main(args.requests, args.questions, args.zipf, args.seed))).read_text(
            encoding="utf-8"
        )
    )
