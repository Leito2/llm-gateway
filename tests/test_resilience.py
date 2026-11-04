"""M3 · Resilience: breaker, retries, timeouts per phase, bulkhead, load shedding, singleflight, shutdown.

Unit tests drive the pieces with fake clocks and seeded RNGs. Integration tests inject fake providers into
an app whose only real provider is `mock`, so nothing touches the network."""

import asyncio
import json
import random

import httpx
import pytest
from fastapi.testclient import TestClient
from helpers import FULL_KEY, make_app, make_client

from llm_gateway.providers.base import ProviderError
from llm_gateway.providers.mock import MockProvider
from llm_gateway.resilience.admission import Admission, effective_priority
from llm_gateway.resilience.breaker import BreakerConfig, BreakerRegistry, CircuitBreaker, State
from llm_gateway.resilience.bulkhead import Bulkhead
from llm_gateway.resilience.lifecycle import Lifecycle
from llm_gateway.resilience.retry import backoff_s
from llm_gateway.resilience.singleflight import Shared, Singleflight, flight_key

URL = "/v1/chat/completions"
BODY = {"model": "fast", "messages": [{"role": "user", "content": "hola"}]}


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


# --- circuit breaker ------------------------------------------------------------------------------


def make_breaker(**cfg):
    clock = FakeClock()
    return CircuitBreaker(BreakerConfig(**cfg), clock), clock


def test_breaker_opens_on_three_consecutive_failures():
    b, _ = make_breaker()
    for _ in range(2):
        b.record_failure()
    assert b.state is State.CLOSED
    b.record_failure()
    assert b.state is State.OPEN and not b.allow()


def test_a_success_resets_the_consecutive_count():
    b, _ = make_breaker()
    for _ in range(2):
        b.record_failure()
    b.record_success()
    for _ in range(2):
        b.record_failure()
    assert b.state is State.CLOSED


def test_breaker_opens_on_error_rate_over_half_once_ten_calls_exist():
    b, _ = make_breaker()
    for i in range(10):  # fail, ok, fail, ok...: never 3 in a row, 50% errors
        (b.record_failure if i % 2 == 0 else b.record_success)()
    assert b.state is State.CLOSED  # 5/10 is not "above 50%"
    b.record_failure()  # 6/11
    assert b.state is State.OPEN


def test_error_rate_needs_the_minimum_number_of_calls():
    b, _ = make_breaker(consecutive_failures=99)
    for i in range(9):
        b.record_failure() if i % 3 else b.record_success()
    assert b.state is State.CLOSED  # 6 errors in 9 calls, but fewer than 10 calls seen


def test_breaker_opens_on_slow_call_rate():
    b, _ = make_breaker()
    for _ in range(6):
        b.record_success(slow=True)
    for _ in range(4):
        b.record_success(slow=False)
    assert b.state is State.CLOSED  # 6/10 is not "above 60%"
    b.record_success(slow=True)  # window slides: still 7 slow of the last 11
    assert b.state is State.OPEN


def test_half_open_lets_one_trial_through_and_closes_on_success():
    b, clock = make_breaker()
    for _ in range(3):
        b.record_failure()
    clock.now = 29.9
    assert not b.allow()
    clock.now = 30.0
    assert b.allow() and b.state is State.HALF_OPEN
    assert not b.allow()  # one trial at a time
    b.record_success()
    assert b.state is State.CLOSED and b.allow()


def test_failed_trial_reopens_with_double_cooldown_capped_at_five_minutes():
    b, clock = make_breaker()
    for _ in range(3):
        b.record_failure()
    expected = [30.0, 60.0, 120.0, 240.0, 300.0, 300.0]
    clock.now = 0.0
    waits = []
    for _ in expected:
        waits.append(b.snapshot()["retry_in_s"])
        clock.now += b.snapshot()["retry_in_s"]
        assert b.allow() and b.state is State.HALF_OPEN
        b.record_failure()
    assert waits == expected


def test_results_that_arrive_while_open_are_ignored_and_a_cancelled_trial_frees_its_slot():
    b, clock = make_breaker()
    for _ in range(3):
        b.record_failure()
    b.record_success()  # a call that began before the breaker opened
    assert b.state is State.OPEN
    clock.now = 30.0
    assert b.allow()
    b.release_trial()  # the trial was cancelled: no verdict
    assert b.allow()


def test_registry_keeps_one_breaker_per_provider_and_reports_state():
    reg = BreakerRegistry(clock=FakeClock())
    assert reg.get("a") is reg.get("a") and reg.get("a") is not reg.get("b")
    assert set(reg.snapshot()) == {"a", "b"} and reg.snapshot()["a"]["state"] == "closed"


# --- retry backoff, bulkhead, admission -----------------------------------------------------------


def test_backoff_is_full_jitter_capped_at_500ms():
    rng = random.Random(7)
    for attempt in range(8):
        for _ in range(50):
            assert 0.0 <= backoff_s(attempt, rng=rng) <= min(0.5, 0.1 * 2**attempt)
    assert max(backoff_s(10, rng=rng) for _ in range(200)) > 0.25  # the cap is reached, not undershot


async def test_bulkhead_limits_concurrency_per_provider():
    b = Bulkhead({"ollama": 2})
    assert await b.acquire("ollama", 0.01) and await b.acquire("ollama", 0.01)
    assert not await b.acquire("ollama", 0.01)  # full: the caller moves on
    assert await b.acquire("groq", 0.01)  # no limit declared
    b.release("ollama")
    assert await b.acquire("ollama", 0.01)
    assert b.snapshot() == {"ollama": {"limit": 2, "free": 0}}


def test_admission_sheds_batch_at_75_percent_and_interactive_at_100():
    adm = Admission(limit=64)
    held = [adm.try_admit("interactive") for _ in range(48)]
    assert adm.try_admit("batch") is None  # 48 of 64: batch is turned away
    held += [adm.try_admit("interactive") for _ in range(16)]
    assert all(held) and adm.try_admit("interactive") is None
    held[0].release()
    held[0].release()  # a ticket releases once, however often it is called
    assert adm.in_flight == 63 and adm.try_admit("interactive") is not None


def test_a_client_can_lower_its_priority_but_never_raise_it():
    assert effective_priority("interactive", "batch") == "batch"
    assert effective_priority("interactive", None) == "interactive"
    assert effective_priority("batch", "interactive") == "batch"
    assert effective_priority("interactive", "garbage") == "interactive"


# --- singleflight ---------------------------------------------------------------------------------


async def test_singleflight_followers_get_the_leaders_result():
    sf = Singleflight()
    lead, flight = sf.join("k")
    follow, same = sf.join("k")
    assert lead and not follow and same is flight and sf.in_flight == 1
    waiter = asyncio.create_task(same.wait(1))
    flight.resolve(Shared({"x": 1}, "mock:m"))
    assert (await waiter).completion == {"x": 1} and sf.in_flight == 0
    assert sf.join("k")[0]  # the key is free again


async def test_a_follower_that_gives_up_does_not_cancel_the_leader():
    sf = Singleflight()
    _, flight = sf.join("k")
    assert await flight.wait(0.01) is None  # timed out
    flight.resolve(Shared({"x": 1}, "mock:m"))  # the leader still resolves normally
    assert (await flight.wait(0.01)).completion == {"x": 1}


def test_flight_key_separates_clients_aliases_and_parameters():
    base = ("c", "judge", [{"role": "user", "content": "x"}], None, 100, 0.0)
    assert flight_key(*base) == flight_key(*base)
    assert (
        len(
            {
                flight_key(*base),
                flight_key("d", *base[1:]),
                flight_key(*base[:5], 0.7),
                flight_key(*base[:4], 50, 0.0),
            }
        )
        == 4
    )


# --- helpers for integration tests ----------------------------------------------------------------


class Counting(MockProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    async def complete(self, request, model):
        self.calls += 1
        return await super().complete(request, model)

    async def stream(self, request, model):
        self.calls += 1
        async for chunk in super().stream(request, model):
            yield chunk


class Failing(Counting):
    def __init__(self, kind="transient", status=503, retry_after=None):
        super().__init__()
        self.err = (kind, status, retry_after)

    def _fail(self):
        self.calls += 1
        kind, status, retry_after = self.err
        raise ProviderError("boom", kind=kind, status=status, retry_after=retry_after)

    async def complete(self, request, model):
        self._fail()

    async def stream(self, request, model):
        self._fail()
        yield {}


class Gate(Counting):
    """Blocks calls for `models` (all models when None) until `open` is set."""

    def __init__(self, models=None, **kw):
        super().__init__(**kw)
        self.open, self.entered, self.models = asyncio.Event(), 0, models

    async def _hold(self, model):
        self.entered += 1
        if self.models is None or model in self.models:
            await self.open.wait()

    async def complete(self, request, model):
        await self._hold(model)
        return await super().complete(request, model)

    async def stream(self, request, model):
        await self._hold(model)
        async for chunk in super().stream(request, model):
            yield chunk


class Stalling(Counting):
    """First chunk arrives, then nothing."""

    async def stream(self, request, model):
        self.calls += 1
        async for chunk in MockProvider.stream(self, request, model):
            yield chunk
            break
        await asyncio.sleep(30)


def app_with(**providers):
    app = make_app()
    app.state.gateway.providers.update(providers)
    return app, app.state.gateway


def aclient(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://t",
        headers={"Authorization": f"Bearer {FULL_KEY}"},
    )


async def until(predicate, limit_s=3.0):
    async with asyncio.timeout(limit_s):
        while not predicate():  # noqa: ASYNC110 (polling a plain predicate)
            await asyncio.sleep(0.01)


def sse_events(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def sse_text(text):
    out = ""
    for ev in sse_events(text):
        if ev != "[DONE]" and not ev.startswith('{"error"'):
            for ch in json.loads(ev)["choices"]:
                out += ch["delta"].get("content") or ""
    return out


# --- fallback, breaker in the request path (G4, R03, R04) -----------------------------------------


def test_g4_a_primary_that_always_fails_costs_no_request_and_the_breaker_opens():
    groq = Failing()
    app, gw = app_with(groq=groq)
    client = TestClient(app, headers=make_client().headers)
    for _ in range(100):
        r = client.post(URL, json=BODY)
        assert r.status_code == 200 and r.headers["x-gateway-provider"].startswith("mock:")
    assert client.get("/stats").json()["breakers"]["groq"]["state"] == "open"
    assert client.get("/healthz").json()["providers"]["groq"]["state"] == "open"


def test_r04_an_open_breaker_is_skipped_without_calling_the_provider():
    groq = Failing()
    app, gw = app_with(groq=groq)
    client = TestClient(app, headers=make_client().headers)
    for _ in range(3):
        client.post(URL, json=BODY)
    calls = groq.calls
    assert gw.breakers.get("groq").state is State.OPEN and calls == 3
    for _ in range(10):
        client.post(URL, json=BODY)
    assert groq.calls == calls


def test_half_open_trial_goes_to_the_real_provider_and_a_success_closes_the_breaker():
    clock = FakeClock()
    groq = Counting()
    app, gw = app_with(groq=groq)
    breaker = gw.breakers._breakers["groq"] = CircuitBreaker(BreakerConfig(), clock)
    for _ in range(3):
        breaker.record_failure()
    client = TestClient(app, headers=make_client().headers)
    assert client.post(URL, json=BODY).headers["x-gateway-provider"].startswith("mock:")
    clock.now = 30.0
    assert client.post(URL, json=BODY).headers["x-gateway-provider"].startswith("groq:")
    assert breaker.state is State.CLOSED


# --- retries --------------------------------------------------------------------------------------


def retried(err_kind, status):
    groq = Failing(err_kind, status)
    app, gw = app_with(groq=groq)
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    gw.executor.sleep = fake_sleep
    r = TestClient(app, headers=make_client().headers).post(URL, json=BODY)
    assert r.status_code == 200
    return groq.calls, sleeps


def test_a_5xx_is_retried_once_with_jitter_then_falls_back():
    calls, sleeps = retried("transient", 503)
    assert calls == 2 and len(sleeps) == 1 and 0 <= sleeps[0] <= 0.1


def test_a_connection_error_is_retried_once():
    calls, _ = retried("transient", None)
    assert calls == 2


@pytest.mark.parametrize(
    ("kind", "status"), [("transient", 429), ("transient", 408), ("provider", 401), ("provider", 404)]
)
def test_429_timeouts_and_provider_errors_are_not_retried(kind, status):
    calls, sleeps = retried(kind, status)
    assert calls == 1 and sleeps == []


def test_a_request_makes_at_most_five_provider_calls():
    google, groq, mock = Failing(), Failing(), Failing()
    app, gw = app_with(google=google, groq=groq, mock=mock)

    async def no_sleep(_):
        pass

    gw.executor.sleep = no_sleep
    r = TestClient(app, headers=make_client().headers).post(URL, json={**BODY, "model": "smart"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_providers_unavailable"
    assert (google.calls, groq.calls, mock.calls) == (2, 2, 1)  # the fifth call is the last


# --- timeouts per phase ---------------------------------------------------------------------------


def test_slow_first_token_falls_back_and_is_not_retried():
    groq = Counting(ttft_s=2.0)
    app, gw = app_with(groq=groq)
    gw.routes.aliases["fast"].ttft_timeout_s = 0.05
    r = TestClient(app, headers=make_client().headers).post(URL, json={**BODY, "stream": True})
    assert r.status_code == 200 and r.headers["x-gateway-provider"].startswith("mock:")
    assert r.text.strip().endswith("[DONE]") and groq.calls == 1


def test_total_timeout_wins_over_every_other_timeout():
    app, gw = app_with(groq=Counting(ttft_s=5.0), mock=Counting(ttft_s=5.0))
    gw.routes.aliases["fast"].total_timeout_s = 0.15
    r = TestClient(app, headers=make_client().headers).post(URL, json=BODY)
    assert r.status_code == 504 and r.json()["error"]["code"] == "upstream_timeout"


def test_a_stream_that_stalls_after_its_first_token_ends_with_a_typed_error_and_no_fallback():
    groq = Stalling()
    app, gw = app_with(groq=groq)
    gw.routes.aliases["fast"].stall_timeout_s = 0.1
    client = TestClient(app, headers=make_client().headers)
    r = client.post(URL, json={**BODY, "stream": True})
    events = sse_events(r.text)
    assert r.status_code == 200 and r.headers["x-gateway-provider"].startswith("groq:")
    assert json.loads(events[-1])["error"]["code"] == "stream_stalled"
    assert "[DONE]" not in events  # an error must not look like a clean end
    assert gw.breakers.get("groq").snapshot()["consecutive_failures"] == 1
    assert gw.admission.in_flight == 0 and gw.bulkhead.snapshot() == {"ollama": {"limit": 2, "free": 2}}


# --- bulkhead and load shedding (R14) -------------------------------------------------------------


async def test_a_full_provider_is_skipped_instead_of_queued():
    groq = Gate()
    app, gw = app_with(groq=groq)
    gw.bulkhead = gw.executor.bulkhead = Bulkhead({"groq": 1})
    async with aclient(app) as c:
        first = asyncio.create_task(c.post(URL, json=BODY))
        await until(lambda: groq.entered == 1)
        second = await c.post(URL, json=BODY)
        assert second.headers["x-gateway-provider"].startswith("mock:")
        groq.open.set()
        assert (await first).headers["x-gateway-provider"].startswith("groq:")
    assert gw.bulkhead.snapshot()["groq"]["free"] == 1


def say(model, n):
    return {"model": model, "messages": [{"role": "user", "content": f"question {n}"}]}


async def test_r14_batch_requests_are_shed_first_and_interactive_only_when_full():
    gate = Gate(models={"mock-judge", "mock-fast"})
    app, gw = app_with(mock=gate)
    gw.admission = Admission(limit=4)  # batch limit: 3
    async with aclient(app) as c:
        held = [asyncio.create_task(c.post(URL, json=say("judge", i))) for i in range(3)]
        await until(lambda: gw.admission.in_flight == 3)
        shed = await c.post(URL, json=say("judge", 99))
        assert shed.status_code == 503 and shed.json()["error"]["code"] == "overloaded"
        assert shed.headers["retry-after"] == "5"
        lowered = await c.post(URL, json=BODY, headers={"X-Priority": "batch"})
        assert lowered.status_code == 503  # `fast` asked to be batch
        held.append(asyncio.create_task(c.post(URL, json=say("fast", 1))))  # interactive: still admitted
        await until(lambda: gw.admission.in_flight == 4)
        full = await c.post(URL, json=say("fast", 2))
        assert full.status_code == 503 and full.headers["retry-after"] == "1"
        gate.open.set()
        assert [r.status_code for r in await asyncio.gather(*held)] == [200] * 4
    assert gw.admission.in_flight == 0


def test_the_slot_and_ticket_come_back_when_a_streaming_client_leaves_early():
    app, gw = app_with(mock=Counting(token_delay_s=0.05))
    client = TestClient(app, headers=make_client().headers)
    words = " ".join(f"w{i}" for i in range(40))
    with client.stream(
        "POST", URL, json={"model": "fast", "stream": True, "messages": [{"role": "user", "content": words}]}
    ) as r:
        next(r.iter_lines())
    deadline = 50
    while gw.admission.in_flight and deadline:
        deadline -= 1
        client.get("/healthz")
    assert gw.admission.in_flight == 0


# --- singleflight in the request path -------------------------------------------------------------


JUDGE = {"model": "judge", "messages": [{"role": "user", "content": "grade this"}]}


async def test_identical_cacheable_requests_in_flight_share_one_provider_call():
    gate = Gate()
    app, gw = app_with(mock=gate)
    async with aclient(app) as c:
        leader = asyncio.create_task(c.post(URL, json=JUDGE))
        await until(lambda: gate.entered == 1)
        follower = asyncio.create_task(c.post(URL, json=JUDGE))
        await until(lambda: gw.admission.in_flight == 2)
        await asyncio.sleep(0.05)
        gate.open.set()
        a, b = await leader, await follower
    assert gate.calls == 1 and a.status_code == b.status_code == 200
    assert b.headers["x-gateway-singleflight"] == "shared" and "x-gateway-singleflight" not in a.headers
    assert a.json()["choices"] == b.json()["choices"] and gw.singleflight.in_flight == 0


async def test_non_cacheable_aliases_never_share_a_call():
    gate = Gate()
    app, gw = app_with(mock=gate)
    async with aclient(app) as c:
        tasks = [asyncio.create_task(c.post(URL, json=BODY)) for _ in range(2)]
        await until(lambda: gate.entered == 2)
        gate.open.set()
        await asyncio.gather(*tasks)
    assert gate.calls == 2


async def test_a_follower_runs_on_its_own_when_the_leader_fails():
    class FailsOnce(Gate):
        async def complete(self, request, model):
            self.entered += 1
            first = self.entered == 1
            if first:
                await self.open.wait()
                raise ProviderError("boom", kind="provider", status=401)
            return await Counting.complete(self, request, model)

    gate = FailsOnce()
    app, gw = app_with(mock=gate)
    async with aclient(app) as c:
        leader = asyncio.create_task(c.post(URL, json=JUDGE))
        await until(lambda: gate.entered == 1)
        follower = asyncio.create_task(c.post(URL, json=JUDGE))
        await until(lambda: gw.admission.in_flight == 2)
        await asyncio.sleep(0.05)
        gate.open.set()
        a, b = await leader, await follower
    assert a.status_code == 503 and b.status_code == 200 and "x-gateway-singleflight" not in b.headers


async def test_a_streaming_follower_gets_a_replay_of_the_leaders_stream():
    gate = Gate()
    app, gw = app_with(mock=gate)
    body = {**JUDGE, "stream": True, "stream_options": {"include_usage": True}}
    async with aclient(app) as c:
        leader = asyncio.create_task(c.post(URL, json=body))
        await until(lambda: gate.entered == 1)
        follower = asyncio.create_task(c.post(URL, json=body))
        await until(lambda: gw.admission.in_flight == 2)
        await asyncio.sleep(0.05)
        gate.open.set()
        a, b = await leader, await follower
    assert gate.calls == 1 and b.headers["x-gateway-singleflight"] == "shared"
    assert sse_text(a.text) == sse_text(b.text) == "Mock reply to: grade this"
    assert sse_events(b.text)[-1] == "[DONE]" and '"usage"' in b.text


# --- graceful shutdown (R23) ----------------------------------------------------------------------


async def test_lifecycle_closes_its_window_after_the_drain_timeout():
    life = Lifecycle(drain_timeout_s=0.05)
    life.attach(asyncio.get_running_loop(), hook_signals=False)
    assert not life.draining and not life.stop.done()
    life.begin_drain()
    life.begin_drain()  # a second SIGTERM changes nothing
    await asyncio.wait_for(life.stop, 1)
    assert life.draining


async def test_r23_after_sigterm_new_requests_get_503_and_readyz_fails():
    app, gw = app_with()
    gw.lifecycle.attach(asyncio.get_running_loop(), hook_signals=False)
    async with aclient(app) as c:
        assert (await c.get("/readyz")).status_code == 200
        gw.lifecycle.begin_drain()
        await until(lambda: gw.lifecycle.draining)
        ready = await c.get("/readyz")
        r = await c.post(URL, json=BODY)
        assert (await c.get("/healthz")).status_code == 200
    assert ready.status_code == 503 and ready.json() == {"status": "draining"}
    assert r.status_code == 503 and r.json()["error"]["code"] == "server_shutting_down"
    assert r.headers["retry-after"] == "5"


async def test_r23_a_stream_that_outlives_the_window_is_cut_with_a_typed_error_and_releases_everything():
    app, gw = app_with(mock=Counting(token_delay_s=0.05))
    gw.lifecycle.drain_timeout_s = 0.2
    gw.lifecycle.attach(asyncio.get_running_loop(), hook_signals=False)
    long_prompt = " ".join(f"w{i}" for i in range(60))  # ~3 s of tokens
    async with aclient(app) as c:
        slow = asyncio.create_task(
            c.post(
                URL,
                json={
                    "model": "fast",
                    "stream": True,
                    "messages": [{"role": "user", "content": long_prompt}],
                },
            )
        )
        await until(lambda: gw.admission.in_flight == 1)
        gw.lifecycle.begin_drain()
        r = await asyncio.wait_for(slow, 2)
    events = sse_events(r.text)
    assert json.loads(events[-1])["error"]["code"] == "server_shutting_down" and "[DONE]" not in events
    assert gw.admission.in_flight == 0


async def test_r23_a_stream_that_finishes_inside_the_window_completes_normally():
    app, gw = app_with(mock=Counting(token_delay_s=0.02))
    gw.lifecycle.drain_timeout_s = 2.0
    gw.lifecycle.attach(asyncio.get_running_loop(), hook_signals=False)
    async with aclient(app) as c:
        slow = asyncio.create_task(c.post(URL, json={**BODY, "stream": True}))
        await until(lambda: gw.admission.in_flight == 1)
        gw.lifecycle.begin_drain()
        r = await asyncio.wait_for(slow, 3)
    assert sse_events(r.text)[-1] == "[DONE]" and sse_text(r.text) == "Mock reply to: hola"
