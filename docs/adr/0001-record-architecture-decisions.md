# ADR-0001: Record architecture decisions

- **Status:** accepted
- **Date:** 2026-10-06

## Context
This project makes many trade-offs (latency vs guarantees, cost vs quality, local vs cloud).
They must be explicit, reviewable and linkable from the README.

## Decision
We record each significant decision as a short ADR in `docs/adr/NNNN-title.md`
(context → decision → consequences). The summaries in `PLAN.md` are the starting set;
each one becomes its own ADR when the corresponding milestone is implemented.

## Consequences
Decisions can be revisited with their original reasoning in view, and interview
explanations point to a written record.
