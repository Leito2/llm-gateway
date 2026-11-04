"""G6: precision-coverage curve and thresholds for one embedder.

Usage: uv run python scripts/calibrate_cache.py <embedder> [pairs.jsonl]

Each line of the pairs file: {"a": "...", "b": "...", "same": true, "lang": "es"}.
`same` = both questions deserve the same answer. Include hard negatives (same words, different fact).
Prints the curve, the chosen thresholds and the YAML to paste into config/cache.yaml. Nothing is written."""

import asyncio
import json
import sys
from pathlib import Path

from llm_gateway.cache import guards
from llm_gateway.cache.calibration import curve, false_hit_rate, pick_t_high, pick_t_low
from llm_gateway.cache.embedder import build_embedder, cosine

DEFAULT_PAIRS = Path("tests/data/pairs_seed.jsonl")


async def main(name: str, pairs: list[dict]) -> int:
    embedder = build_embedder(name)
    if embedder is None:
        print(f"No embedder named {name!r} is available here (model files missing or not implemented).")
        return 2
    scored = []
    for p in pairs:
        va, vb = await embedder.embed([p["a"], p["b"]])
        scored.append((cosine(va, vb), bool(p["same"]), guards.mismatch(p["a"], p["b"]) is None))
    points = curve(scored)
    positives = sum(1 for _, same, _ in scored if same)
    print(f"{embedder.name}: {len(scored)} pairs ({positives} same, {len(scored) - positives} different)\n")
    print("threshold  served  false  precision  coverage")
    for pt in points[:: max(1, len(points) // 15)]:
        row = (pt.threshold, pt.served, pt.false_hits, pt.precision, pt.coverage)
        print("  {:.3f}    {:5d}  {:5d}   {:7.3f}   {:7.3f}".format(*row))
    high = pick_t_high(points)
    if high is None:
        print("\nNo threshold reaches 99.5% precision on this set: keep the layer in observation mode.")
        return 1
    low = pick_t_low(scored, high.threshold)
    print(
        f"\nt_high={high.threshold:.3f}  coverage={high.coverage:.1%}  "
        f"false-hit rate={false_hit_rate(points, high.threshold):.2%}  t_low={low:.3f}"
    )
    print(f"\n  {name}: {{ t_high: {high.threshold:.3f}, t_low: {low:.3f} }}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    pairs_file = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PAIRS
    lines = pairs_file.read_text(encoding="utf-8").splitlines()
    sys.exit(asyncio.run(main(sys.argv[1], [json.loads(line) for line in lines if line.strip()])))
