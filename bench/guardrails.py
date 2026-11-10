"""G14: recall and false-positive rate of the injection / PII heuristics in guardrails/scan.py.

    uv run python -m bench.guardrails

Data: tests/data/guardrail_eval.jsonl, written by hand for this project (NOT a public benchmark; PLAN asks for
one). It deliberately includes hard positives (paraphrases, spaced-out or leetspeak text, indirect requests)
and tricky benign prompts that contain trigger words ("ignore a failing test", "developer mode on Android").
Numbers describe THIS set only; a regex stand-in is expected to miss paraphrases a model would catch."""

import json
import sys

from bench.common import ROOT, save
from llm_gateway.api.schemas import ChatMessage
from llm_gateway.guardrails.scan import scan

HARD_FROM = 18  # the first 18 injection rows are direct phrases, the rest are paraphrase/obfuscation


def rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def main() -> dict:
    rows = [
        json.loads(line)
        for line in (ROOT / "tests/data/guardrail_eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    inj_pos = [r for r in rows if r["injection"]]
    inj_neg = [r for r in rows if not r["injection"]]
    found = {id(r): scan([ChatMessage(role="user", content=r["text"])]) for r in rows}

    direct, hard = inj_pos[:HARD_FROM], inj_pos[HARD_FROM:]
    hits_direct = sum(found[id(r)].injection for r in direct)
    hits_hard = sum(found[id(r)].injection for r in hard)
    false_pos = [r["text"] for r in inj_neg if found[id(r)].injection]
    missed = [r["text"] for r in hard if not found[id(r)].injection]

    kinds = {"tp": 0, "fn": 0, "fp": 0}
    pii_missed, pii_false = [], []
    for r in rows:
        want, got = set(r["pii"]), set(found[id(r)].pii)
        kinds["tp"] += len(want & got)
        kinds["fn"] += len(want - got)
        kinds["fp"] += len(got - want)
        if want - got:
            pii_missed.append((r["text"], sorted(want - got)))
        if got - want:
            pii_false.append((r["text"], sorted(got - want)))
    return {
        "dataset": {
            "injection_positive": len(inj_pos),
            "injection_negative": len(inj_neg),
            "rows": len(rows),
        },
        "injection": {
            "recall_direct_phrases": rate(hits_direct, len(direct)),
            "recall_paraphrase_and_obfuscation": rate(hits_hard, len(hard)),
            "recall_overall": rate(hits_direct + hits_hard, len(inj_pos)),
            "false_positive_rate": rate(len(false_pos), len(inj_neg)),
            "false_positives": false_pos,
            "missed": missed,
        },
        "pii": {
            "recall": rate(kinds["tp"], kinds["tp"] + kinds["fn"]),
            "precision": rate(kinds["tp"], kinds["tp"] + kinds["fp"]),
            "missed": pii_missed,
            "false_positives": pii_false,
        },
        "note": (
            "hand-written set; regex heuristic only (the Prompt Guard 2 classifier has not run on real "
            "weights; Presidio redacts logs and is not scored here)"
        ),
    }


if __name__ == "__main__":
    result = main()
    print(save("g14_guardrails", result).read_text(encoding="utf-8"))
    sys.exit(0)
