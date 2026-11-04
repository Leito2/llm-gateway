"""G6 harness: the threshold-picking maths, and the seed pairs as a smoke test (not a quality claim)."""

import json
from pathlib import Path

from llm_gateway.cache import guards
from llm_gateway.cache.calibration import curve, false_hit_rate, pick_t_high, pick_t_low
from llm_gateway.cache.embedder import HashEmbedder, cosine

# (similarity, same question?, guards pass?)
SCORED = [
    (0.95, True, True), (0.93, True, True), (0.91, True, True), (0.88, True, True),
    (0.86, False, True),                                 # a false hit that sits inside the true ones
    (0.84, True, True), (0.80, True, True), (0.74, True, True),
    (0.70, False, True), (0.60, False, True), (0.55, True, True),
    (0.97, False, False),                                # very close but a guard rejects it: never served
]  # fmt: skip


def test_curve_counts_only_pairs_that_pass_the_guards():
    pts = {p.threshold: p for p in curve(SCORED)}
    assert pts[0.97].served == 0 and pts[0.97].precision == 1.0
    top = pts[0.88]
    assert (top.served, top.false_hits, top.precision) == (4, 0, 1.0) and top.coverage == 4 / 8
    assert pts[0.86].false_hits == 1 and pts[0.86].served == 5


def test_t_high_is_the_loosest_threshold_that_meets_the_precision_target():
    points = curve(SCORED)
    assert pick_t_high(points, 0.995).threshold == 0.88  # 0.86 lets the false hit in
    assert pick_t_high(points, 0.80).threshold == 0.74  # a looser target buys coverage
    assert pick_t_high(curve([(0.9, False, True)]), 0.995) is None


def test_t_low_keeps_the_requested_share_of_true_pairs_and_never_exceeds_t_high():
    assert pick_t_low(SCORED, 0.88, recall=1.0) == 0.55
    assert pick_t_low(SCORED, 0.88, recall=0.75) == 0.80
    assert pick_t_low(SCORED, 0.70, recall=0.10) == 0.70  # capped at t_high
    assert pick_t_low([(0.5, False, True)], 0.9) == 0.9


def test_false_hit_rate_is_wrong_hits_over_served_hits():
    points = curve(SCORED)
    assert false_hit_rate(points, 0.88) == 0.0
    assert false_hit_rate(points, 0.86) == 1 / 5
    assert false_hit_rate(points, 0.123) == 0.0  # unknown threshold


async def test_seed_pairs_run_through_the_pipeline():
    path = Path(__file__).parent / "data" / "pairs_seed.jsonl"
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(pairs) >= 40 and {p["lang"] for p in pairs} == {"en", "es"}
    assert 0.4 < sum(p["same"] for p in pairs) / len(pairs) < 0.6  # balanced
    e = HashEmbedder()
    scored = []
    for p in pairs:
        va, vb = await e.embed([p["a"], p["b"]])
        scored.append((cosine(va, vb), p["same"], guards.mismatch(p["a"], p["b"]) is None))
    # Negatives that differ in a number, date, code or negation are stopped by the guards, not the embedder.
    guarded = [p for p, (_, _, ok) in zip(pairs, scored, strict=True) if not ok]
    assert guarded and all(not p["same"] for p in guarded)
