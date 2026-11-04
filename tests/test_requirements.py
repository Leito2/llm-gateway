"""EARS requirements that had no dedicated end-to-end test of their own (PLAN §11). The README maps every
requirement R01-R23 to the test that proves it; this file holds R05, R06, R12 and R20."""

import asyncio

import httpx
from fastapi.testclient import TestClient
from helpers import FULL_KEY, make_app
from test_anthropic import MESSAGE, provider
from test_budget import POOR_KEY, RICH_KEY, _app, _post

from llm_gateway.budget.memory import MemoryBudget
from llm_gateway.budget.store import BudgetUnavailable, Caps, Scopes


def test_req_05_no_paid_call_without_an_accepted_reservation_and_402_when_nothing_is_left():
    app, paid = _app()
    with TestClient(app) as c:
        skipped = _post(c, POOR_KEY, "paid")  # client cap 0: reservation refused, chain goes on
        nothing_left = _post(c, POOR_KEY, "onlypaid")
    assert paid.calls == 0  # the provider was never called without a reservation
    assert skipped.status_code == 200 and skipped.headers["x-budget-skipped"] == "anthropic"
    assert nothing_left.status_code == 402 and nothing_left.json()["error"]["code"] == "budget_exceeded"


async def test_req_06_settled_spend_never_exceeds_the_cap_at_any_concurrency():
    store, cap, cost = MemoryBudget(), 1000, 90
    scopes, caps = Scopes("r", "c", "g"), Caps(run=cap, client_day=None, global_day=None)

    async def one(i: int) -> int:
        if not await store.reserve(f"id{i}", cost, 60, scopes, caps):
            return 0
        await asyncio.sleep(0)  # let other requests interleave between reserve and settle
        await store.settle(f"id{i}", cost, scopes)
        return cost

    spent = sum(await asyncio.gather(*(one(i) for i in range(200))))
    snap = (await store.snapshot(scopes))["run"]
    assert spent == snap["spent"] <= cap and snap["reserved"] == 0


def test_req_12_budget_store_down_rejects_paid_providers_and_free_ones_keep_serving():
    class Down:
        async def reserve(self, *a, **k):
            raise BudgetUnavailable("redis is down")

        async def settle(self, *a, **k):
            raise BudgetUnavailable("redis is down")

        async def snapshot(self, *a, **k):
            raise BudgetUnavailable("redis is down")

    app, paid = _app()
    app.state.gateway.budget.store = Down()
    with TestClient(app) as c:
        r = _post(c, RICH_KEY, "paid")  # this client has budget, but the store cannot confirm it
    assert paid.calls == 0 and r.status_code == 200
    assert r.headers["x-budget-skipped"] == "anthropic" and r.headers["x-gateway-provider"].startswith("mock")


def test_req_20_a_refusal_is_content_filter_with_no_fallback_and_is_never_cached():
    calls = []

    def refuse(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={**MESSAGE, "stop_reason": "refusal"})

    app = make_app(run_budget_usd=1.0, cache_backend="memory")
    anthropic, _ = provider(refuse)
    app.state.gateway.providers["anthropic"] = anthropic
    body = {"model": "smart_paid", "messages": [{"role": "user", "content": "refuse me"}], "cache": True}
    auth = {"Authorization": f"Bearer {FULL_KEY}"}
    with TestClient(app) as c:
        first = c.post("/v1/chat/completions", headers=auth, json=body)
        for _ in range(50):  # let any (wrongly) pending write-back run
            c.get("/healthz")
        second = c.post("/v1/chat/completions", headers=auth, json=body)
    assert first.json()["choices"][0]["finish_reason"] == "content_filter"
    assert (
        first.headers["x-gateway-provider"].startswith("anthropic") and "x-cache-layer" not in second.headers
    )
    assert len(calls) == 2  # the second request went to the provider again: nothing was cached
