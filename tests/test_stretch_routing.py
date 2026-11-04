"""Stretch · hedging, EWMA reordering and the adaptive bulkhead."""

import asyncio

from fastapi.testclient import TestClient
from helpers import make_client
from test_resilience import Counting, Failing, aclient, app_with

from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.routing.config import Target
from llm_gateway.routing.latency import MIN_SAMPLES_FOR_HEDGE, LatencyTracker

URL = "/v1/chat/completions"
BODY = {"model": "fast", "messages": [{"role": "user", "content": "hola"}]}


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


# --- latency tracker ------------------------------------------------------------------------------


def test_ewma_moves_toward_new_samples_and_counts_them():
    t = LatencyTracker(alpha=0.5)
    t.record("a", 1.0)
    t.record("a", 3.0)
    assert t.ewma("a") == 2.0 and t.samples("a") == 2 and t.ewma("b") is None


def test_reorder_puts_fast_first_unmeasured_before_measured_and_degraded_last():
    t = LatencyTracker()
    t.record("slow", 2.0)
    t.record("quick", 0.1)
    chain = [
        Target(provider="slow", model="m"),
        Target(provider="mock", model="m", degraded=True),
        Target(provider="quick", model="m"),
        Target(provider="new", model="m"),
    ]
    assert [x.provider for x in t.reorder(chain)] == ["new", "quick", "slow", "mock"]


def test_executor_learns_latency_from_successful_calls():
    app, gw = app_with(groq=Counting(ttft_s=0.05))
    client = TestClient(app, headers=make_client().headers)
    assert client.post(URL, json=BODY).status_code == 200
    assert gw.latency.samples("groq") == 1 and gw.latency.ewma("groq") >= 0.04
    assert client.get("/stats").json()["latency"]["groq"]["samples"] == 1


def test_ewma_reorder_sends_the_request_to_the_faster_link_first():
    app, gw = app_with(groq=Counting(ttft_s=0.2), mock=Counting())
    gw.routes.aliases["fast"].chain[-1].degraded = False  # make the mock link a regular one
    gw.routes.aliases["fast"].reorder = "ewma"
    gw.latency.record("groq", 5.0)
    gw.latency.record("mock", 0.01)
    r = TestClient(app, headers=make_client().headers).post(URL, json=BODY)
    assert r.headers["x-gateway-provider"].startswith("mock:")


# --- hedging --------------------------------------------------------------------------------------


def hedged_app(primary, secondary, delay=0.05):
    app, gw = app_with(groq=primary, mock=secondary)
    policy = gw.routes.aliases["fast"]
    policy.hedge, policy.hedge_delay_s = True, delay
    events: list[str] = []
    gw.executor.on_hedge = events.append
    return app, gw, events


async def test_a_slow_primary_loses_to_the_hedge():
    groq, mock = Counting(ttft_s=1.0), Counting()
    app, gw, events = hedged_app(groq, mock)
    async with aclient(app) as c:
        started = asyncio.get_running_loop().time()
        r = await c.post(URL, json=BODY)
        took = asyncio.get_running_loop().time() - started
    assert r.status_code == 200 and r.headers["x-gateway-provider"].startswith("mock:")
    assert took < 0.8  # did not wait for the slow primary
    assert events == ["launched", "won"] and mock.calls == 1
    assert gw.bulkhead.snapshot() == {"ollama": {"limit": 2, "free": 2}}  # the loser's slot came back


async def test_a_fast_primary_never_launches_the_hedge():
    groq, mock = Counting(), Counting()
    app, gw, events = hedged_app(groq, mock, delay=0.5)
    async with aclient(app) as c:
        r = await c.post(URL, json=BODY)
    assert r.headers["x-gateway-provider"].startswith("groq:")
    assert events == [] and mock.calls == 0


async def test_the_primary_still_wins_when_it_answers_after_the_delay_but_first():
    groq, mock = Counting(ttft_s=0.12), Counting(ttft_s=1.0)
    app, gw, events = hedged_app(groq, mock, delay=0.05)
    async with aclient(app) as c:
        r = await c.post(URL, json=BODY)
    assert r.headers["x-gateway-provider"].startswith("groq:")
    assert events == ["launched", "lost"]


async def test_a_hedge_that_wins_marks_the_primary_slow_but_not_failed():
    groq, mock = Counting(ttft_s=1.0), Counting()
    app, gw, _ = hedged_app(groq, mock)
    async with aclient(app) as c:
        await c.post(URL, json=BODY)
    snap = gw.breakers.get("groq").snapshot()
    assert snap["consecutive_failures"] == 0


async def test_a_primary_that_fails_fast_falls_back_without_a_race():
    groq, mock = Failing(), Counting()
    app, gw, events = hedged_app(groq, mock, delay=0.5)

    async def no_sleep(_):
        pass

    gw.executor.sleep = no_sleep
    async with aclient(app) as c:
        r = await c.post(URL, json=BODY)
    assert r.status_code == 200 and r.headers["x-gateway-provider"].startswith("mock:")
    assert events == []


async def test_when_both_racers_fail_the_request_is_an_ordinary_503():
    groq, mock = Failing(), Failing()
    app, gw, events = hedged_app(groq, mock, delay=0.0)

    async def no_sleep(_):
        pass

    gw.executor.sleep = no_sleep
    async with aclient(app) as c:
        r = await c.post(URL, json=BODY)
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_providers_unavailable"


async def test_streaming_hedge_cancels_the_loser_and_streams_the_winner():
    groq, mock = Counting(ttft_s=1.0), Counting()
    app, gw, events = hedged_app(groq, mock)
    async with aclient(app) as c:
        r = await c.post(URL, json={**BODY, "stream": True})
    assert r.status_code == 200 and "[DONE]" in r.text
    assert r.headers["x-gateway-provider"].startswith("mock:") and events == ["launched", "won"]
    assert gw.admission.in_flight == 0


def test_no_hedge_delay_means_no_hedge_until_the_provider_has_history():
    app, gw = app_with(groq=Counting(), mock=Counting())
    policy = gw.routes.aliases["fast"]
    policy.hedge = True
    chain = policy.chain
    assert gw.executor._hedge_delay(policy, chain) is None
    for _ in range(MIN_SAMPLES_FOR_HEDGE):
        gw.latency.record("groq", 0.1)
    delay = gw.executor._hedge_delay(policy, chain)
    assert 0.29 < delay < 0.31  # three typical latencies
    policy.hedge = False
    assert gw.executor._hedge_delay(policy, chain) is None


def test_hedging_metric_is_exposed():
    app, gw, _ = hedged_app(Counting(ttft_s=1.0), Counting())
    gw.executor.on_hedge = gw.metrics.hedges_hook

    async def go():
        async with aclient(app) as c:
            await c.post(URL, json=BODY)

    asyncio.run(go())
    text = TestClient(app, headers=make_client().headers).get("/metrics").text
    assert 'gw_hedges_total{outcome="won"} 1.0' in text and "gw_ttft_ewma_seconds" in text


# --- adaptive bulkhead (AIMD) ---------------------------------------------------------------------


def test_aimd_halves_on_congestion_at_most_once_a_second_and_never_below_the_floor():
    clock = Clock()
    clock.now = 10.0
    b = Bulkhead({"p": 8}, adaptive={"p": 2}, clock=clock)
    b.feedback("p", True)
    assert b.snapshot()["p"]["limit"] == 4
    b.feedback("p", True)  # same second: ignored
    assert b.snapshot()["p"]["limit"] == 4
    clock.now = 12.0
    b.feedback("p", True)
    clock.now = 14.0
    b.feedback("p", True)
    assert b.snapshot()["p"]["limit"] == 2


def test_aimd_grows_by_one_after_a_full_window_of_successes_up_to_the_maximum():
    clock = Clock()
    clock.now = 10.0
    b = Bulkhead({"p": 4}, adaptive={"p": 1}, clock=clock)
    b.feedback("p", True)
    b.feedback("p", True)
    assert b.snapshot()["p"]["limit"] == 2
    for _ in range(2):
        b.feedback("p", False)
    assert b.snapshot()["p"]["limit"] == 3
    for _ in range(20):
        b.feedback("p", False)
    assert b.snapshot()["p"]["limit"] == 4 and b.snapshot()["p"]["max"] == 4


def test_fixed_providers_ignore_feedback():
    b = Bulkhead({"p": 3})
    b.feedback("p", True)
    assert b.snapshot() == {"p": {"limit": 3, "free": 3}}


async def test_a_429_shrinks_an_adaptive_provider_through_the_executor():
    app, gw = app_with(groq=Failing(status=429, retry_after=1), mock=Counting())
    gw.bulkhead = gw.executor.bulkhead = Bulkhead({"groq": 8}, adaptive={"groq": 1})
    async with aclient(app) as c:
        r = await c.post(URL, json=BODY)
    assert r.status_code == 200
    assert gw.bulkhead.snapshot()["groq"]["limit"] == 4
