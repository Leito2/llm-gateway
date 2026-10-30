# ADR-0006: P3 final-test deployment — Cloud Run (internal) + Redis Cloud free + double spend limit

- **Status:** accepted
- **Date:** 2026-10-29

## Context
For P3's final test the gateway runs on Google Cloud Run (managed containers, scale to zero,
monthly free tier: 2M requests, 180,000 vCPU-seconds, 360,000 GiB-seconds; a billing account is
required). The gateway needs Redis for the budget and the cache. Options:
- **Memorystore:** about US$30+/month. Breaks the $0 goal.
- **Redis as a sidecar in the same Cloud Run service:** $0, but Cloud Run scales to zero when idle,
  and every scale-down erases the recorded spend. The run cap would reset and could be spent again.
- **Redis Cloud free plan:** $0, managed and persistent across idle periods; 30 MB and 30
  connections. A third party stores cached prompts and answers.

The original plan also said "internal ingress" while a public UI demo was discussed; both cannot hold.

## Decision
- **Redis Cloud free plan**, in the same region as the Cloud Run service. The final test uses only
  synthetic or public data, so the third-party storage is acceptable under the data policy.
- **Double spend limit:** the gateway budget cap **and** a spend limit in the Anthropic console
  (e.g. US$5). Also a budget alert in Google Cloud.
- **Cloud Run:** at most 1 instance, **internal ingress** (only P3 calls it), keys in Secret Manager.
- **No public demo:** the README shows a GIF/video recorded locally; the `demo` client in
  `clients.example.yaml` is for people who run the project on their own machine.
- After the run, recorded spend is compared with the Anthropic console.

## Consequences
- The budget survives idle periods and restarts at $0 cost.
- About 7,000 cache entries fit in 30 MB, enough for a final test.
- To verify in M9: that the free plan includes the Query Engine (vector search). If it does not,
  the budget stays on Redis Cloud and the semantic cache is disabled for that test.
- One more account to manage.
