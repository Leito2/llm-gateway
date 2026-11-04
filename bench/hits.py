"""G9: latency of a cache hit, exact and semantic, against a real gateway process (HTTP, no cache bypass).

    uv run python -m bench.hits [--embedder e5-small] [--threshold 0.85]

Sequential requests (one at a time), so the figure is the cost of the hit itself, not queueing. The semantic
threshold is set ONLY for this measurement (a temporary cache.yaml); it is not a calibrated value (G6 is still
open), so what this says is how fast a hit is, not how often a right one is found. Pairs come from
tests/data/pairs_seed.jsonl (`same: true`): the first wording is stored, the second asked. Several prefixes
make several copies of each pair so the sample is not tiny; a prefixed copy is itself a semantic neighbour of
the earlier ones, so only the first copy of each pair is a true miss / exact hit (small n for those two).
Target (PLAN): semantic p95 < 60 ms."""

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import httpx
import yaml

from bench.common import KEY, ROOT, body, gateway_stack, percentiles, save

PREFIXES = [
    "",
    "Please: ",
    "Quick question: ",
    "Hello, ",
    "Sorry to bother you, ",
    "Urgent: ",
    "Hi team, ",
    "FYI: ",
]
AUTH = {"Authorization": f"Bearer {KEY}"}


def load_pairs() -> list[tuple[str, str]]:
    rows = [
        json.loads(line)
        for line in (ROOT / "tests/data/pairs_seed.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [(r["a"], r["b"]) for r in rows if r["same"]]


async def timed(client: httpx.AsyncClient, alias: str, text: str) -> tuple[float, str | None]:
    t0 = time.perf_counter()
    r = await client.post("/v1/chat/completions", json=body(alias, text))
    ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text
    return ms, r.headers.get("x-cache-layer")


async def main(embedder: str, threshold: float) -> dict:
    pairs = load_pairs()
    with tempfile.TemporaryDirectory(prefix="gw-hits-") as tmp:
        cfg = Path(tmp) / "cache.yaml"
        cfg.write_text(
            yaml.safe_dump({"embedders": {embedder: {"t_high": threshold, "t_low": threshold}}}),
            encoding="utf-8",
        )
        with gateway_stack(embedder=embedder, extra_env={"GW_CACHE_FILE": str(cfg)}) as (gw, _up):
            async with httpx.AsyncClient(base_url=gw, headers=AUTH, timeout=30) as c:
                for _ in range(20):  # warm-up: connection, JIT, ONNX session
                    await timed(c, "bench", "warm up")
                    await timed(c, "bench_sem", f"warm up question {_}")
                miss, exact, semantic, other = [], [], [], []
                for prefix in PREFIXES:
                    for a, _b in pairs:
                        ms, layer = await timed(
                            c, "bench_sem", prefix + a
                        )  # first time: a miss that gets stored
                        (miss if layer is None else other).append(ms)
                    await asyncio.sleep(0.5)  # write-backs (with an embedding each) finish
                    for a, b in pairs:
                        ms, layer = await timed(c, "bench_sem", prefix + a)
                        (exact if layer == "exact" else other).append(ms)
                        ms, layer = await timed(c, "bench_sem", prefix + b)
                        (semantic if layer == "semantic" else other).append(ms)
    asked = len(PREFIXES) * len(pairs)
    sem = percentiles(semantic)
    return {
        "embedder": embedder,
        "threshold_for_this_run_only": threshold,
        "pairs": len(pairs),
        "paraphrases_asked": asked,
        "paraphrases_served_semantic": len(semantic),
        "miss_with_semantic_lookup_ms": percentiles(miss),
        "exact_hit_ms": percentiles(exact),
        "semantic_hit_ms": sem,
        "pass_semantic_p95_under_60ms": bool(sem) and sem["p95"] < 60,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="e5-small")
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()
    print(save("g9_hits", asyncio.run(main(args.embedder, args.threshold))).read_text(encoding="utf-8"))
