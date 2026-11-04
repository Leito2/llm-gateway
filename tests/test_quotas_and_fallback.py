"""Outbound quotas (R13, G15) and the sequential fallback of M2."""
from fastapi.testclient import TestClient
from helpers import make_app, make_client

from llm_gateway.providers.base import ProviderError
from llm_gateway.providers.mock import MockProvider
from llm_gateway.quotas.limiter import QuotaLimiter, QuotaSpec

BODY = {"model": "fast", "messages": [{"role": "user", "content": "hola"}]}


class Failing(MockProvider):
    def __init__(self, kind, status=None, retry_after=None):
        super().__init__()
        self.kind, self.status, self.retry_after, self.calls = kind, status, retry_after, 0

    def _fail(self):
        self.calls += 1
        raise ProviderError("groq: boom", kind=self.kind, status=self.status, retry_after=self.retry_after)

    async def complete(self, request, model):
        self._fail()

    async def stream(self, request, model):
        self._fail()
        yield {}


def app_with(groq: MockProvider, quotas: dict | None = None):
    """`fast` = groq -> ollama (unregistered here) -> ... -> mock (degraded, allowed for `fast`)."""
    app = make_app()
    gw = app.state.gateway
    gw.providers["groq"] = groq
    if quotas:
        gw.quotas = QuotaLimiter(quotas)
    return app, TestClient(app, headers=make_client().headers)


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


def test_limiter_rpm_refills_over_a_minute():
    clock = FakeClock()
    q = QuotaLimiter({"p": QuotaSpec(rpm=2)}, clock=clock)
    assert [q.try_acquire("p", 1) for _ in range(3)] == [True, True, False]
    clock.now = 30.0           # half a minute gives back one request
    assert q.try_acquire("p", 1) and not q.try_acquire("p", 1)


def test_limiter_tpm_takes_nothing_when_a_limit_fails():
    clock = FakeClock()
    q = QuotaLimiter({"p": QuotaSpec(rpm=10, tpm=100)}, clock=clock)
    assert q.try_acquire("p", 80)
    assert not q.try_acquire("p", 80)        # tpm short
    assert q.try_acquire("p", 20)            # the failed attempt did not consume a request or tokens


def test_limiter_rpd_resets_on_a_new_utc_day():
    day = ["2026-11-03"]
    q = QuotaLimiter({"p": QuotaSpec(rpd=1)}, today=lambda: day[0])
    assert q.try_acquire("p", 1) and not q.try_acquire("p", 1)
    day[0] = "2026-11-04"
    assert q.try_acquire("p", 1)


def test_limiter_block_expires_and_unlimited_providers_always_pass():
    clock = FakeClock()
    q = QuotaLimiter({"p": QuotaSpec(rpm=5)}, clock=clock)
    q.block("p", 10)
    assert not q.try_acquire("p", 1)
    clock.now = 11.0
    assert q.try_acquire("p", 1) and q.try_acquire("mock", 10**9)


def test_req_13_g15_exhausted_quota_routes_to_the_next_provider_without_a_429():
    groq = MockProvider()
    _, client = app_with(groq, {"groq": QuotaSpec(rpm=2)})
    providers = []
    for _ in range(5):
        r = client.post("/v1/chat/completions", json=BODY)
        assert r.status_code == 200
        providers.append(r.headers["x-gateway-provider"].split(":")[0])
    assert providers == ["groq", "groq", "mock", "mock", "mock"]


def test_req_03_transient_failure_falls_back_and_reports_the_depth():
    groq = Failing("transient", status=503)
    _, client = app_with(groq)
    r = client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200 and r.headers["x-gateway-provider"].startswith("mock:")
    assert r.headers["x-gateway-fallback-depth"] == "2" and groq.calls == 1


def test_provider_error_kind_also_falls_back_and_streaming_does_too():
    groq = Failing("provider", status=401)
    _, client = app_with(groq)
    r = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200 and r.headers["x-gateway-provider"].startswith("mock:")
    assert r.text.strip().endswith("data: [DONE]")


def test_client_error_goes_back_to_the_client_and_no_other_provider_is_tried():
    groq = Failing("client", status=400)
    app, client = app_with(groq)
    r = client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    assert groq.calls == 1 and ("fast", "mock", 200) not in {
        (x["alias"], x["provider"], x["status"]) for x in app.state.gateway.stats.snapshot()["requests"]}


def test_a_429_blocks_the_provider_for_later_requests():
    groq = Failing("transient", status=429, retry_after=60)
    _, client = app_with(groq)
    for _ in range(3):
        assert client.post("/v1/chat/completions", json=BODY).status_code == 200
    assert groq.calls == 1                      # the second and third request never tried it


def test_503_when_every_link_fails_and_it_names_the_last_error_but_not_upstream_text():
    app = make_app()
    gw = app.state.gateway
    gw.providers["mock"] = Failing("transient", status=500)
    r = TestClient(app, headers=make_client().headers).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_providers_unavailable"
    assert "groq: boom" in r.json()["error"]["message"]
