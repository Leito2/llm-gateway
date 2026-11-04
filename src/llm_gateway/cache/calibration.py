"""Threshold calibration for G6 (PLAN §6.3 point 1): from labeled pairs to `t_high` and `t_low`.

A *scored pair* is `(similarity, same_question, guards_pass)`. A pair is served as a hit when its similarity
reaches the threshold AND the lexical guards pass, so pairs the guards reject never count as false hits."""

from dataclasses import dataclass

ScoredPair = tuple[float, bool, bool]


@dataclass(frozen=True)
class Point:
    threshold: float
    served: int  # pairs that would be served
    false_hits: int  # served pairs that are NOT the same question
    precision: float  # 1 - false_hits / served (1.0 when nothing is served)
    coverage: float  # served true pairs / all true pairs


def curve(pairs: list[ScoredPair]) -> list[Point]:
    """One point per distinct similarity, from the strictest threshold to the loosest."""
    positives = sum(1 for _, same, _ in pairs if same)
    points = []
    for t in sorted({s for s, _, _ in pairs}, reverse=True):
        served = [(same) for s, same, ok in pairs if s >= t and ok]
        bad = sum(1 for same in served if not same)
        good = len(served) - bad
        points.append(
            Point(
                t,
                len(served),
                bad,
                1.0 - bad / len(served) if served else 1.0,
                good / positives if positives else 0.0,
            )
        )
    return points


def pick_t_high(points: list[Point], min_precision: float = 0.995) -> Point | None:
    """The loosest threshold that still meets the precision target: the most coverage at 99.5% precision."""
    ok = [p for p in points if p.served and p.precision >= min_precision]
    return min(ok, key=lambda p: p.threshold) if ok else None


def pick_t_low(pairs: list[ScoredPair], t_high: float, recall: float = 0.98) -> float:
    """Bottom of the gray zone: the highest threshold that still keeps `recall` of the true pairs above it
    (those below it are lost to the verifier). Never above `t_high`."""
    true_scores = sorted((s for s, same, _ in pairs if same), reverse=True)
    if not true_scores:
        return t_high
    keep = max(1, int(len(true_scores) * recall + 0.9999))
    return min(t_high, true_scores[min(keep, len(true_scores)) - 1])


def false_hit_rate(points: list[Point], threshold: float) -> float:
    """Fraction of served pairs that are wrong at `threshold` (the G6 figure, target < 1%)."""
    p = next((p for p in points if p.threshold == threshold), None)
    return 0.0 if p is None or not p.served else p.false_hits / p.served
