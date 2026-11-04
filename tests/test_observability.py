"""M8: metrics, access log, tracing, admin key, rate limit, size guardrail, prompt-logging rule (R16, R22)."""

import hashlib
import json
import logging
import sys
import types

import fakeredis
import pytest
import yaml
from fastapi.testclient import TestClient
from helpers import CLIENTS_FILE, FULL_KEY, PROVIDERS_FILE, make_app
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_gateway.main import create_app
from llm_gateway.security.ratelimit import RateLimiter
from llm_gateway.settings import Settings
from llm_gateway.telemetry.privacy import PromptLoggingUnavailable

AUTH = {"Authorization": f"Bearer {FULL_KEY}"}
BODY = {"model": "fast", "messages": [{"role": "user", "content": "hello there"}]}
ADMIN_KEY = "admin-" + "a" * 32
ADMIN = {"Authorization": f"Bearer {ADMIN_KEY}"}
ADMIN_HASH = hashlib.sha256(ADMIN_KEY.encode()).hexdigest()


def cached_app():
    app = make_app(cache_backend="memory")
    gw = app.state.gateway
    gw.routes.aliases["fast"].chain[-1].degraded = False  # degraded answers are never stored
    return app, gw


def settle(client, gw):
    """Write-back is a fire-and-forget task on the app's loop; more requests let it run."""
    for _ in range(50):
        client.get("/healthz")
        if not gw.cache._pending:
            return
    raise AssertionError("write-back never finished")


def chat(client, **extra):
    return client.post("/v1/chat/completions", headers=AUTH, json={**BODY, **extra})


# --- metrics ------------------------------------------------------------------------------------


def test_metrics_expose_requests_tokens_and_breakers():
    with TestClient(make_app()) as c:
        assert chat(c).status_code == 200
        assert chat(c, model="nope").status_code == 404
        text = c.get("/metrics").text
    assert 'gw_requests_total{alias="fast",provider="mock",status="200"} 1.0' in text
    assert 'gw_requests_total{alias="unknown",provider="-",status="404"} 1.0' in text
    assert 'gw_tokens_total{direction="input",provider="mock"}' in text
    assert "gw_request_seconds_count" in text and "gw_ttft_seconds_bucket" in text
    assert 'gw_breaker_state{provider="mock"} 0.0' in text


def test_metrics_never_carry_prompt_text():
    with TestClient(make_app()) as c:
        chat(c, messages=[{"role": "user", "content": "super-secret-prompt-text"}])
        assert "super-secret-prompt-text" not in c.get("/metrics").text


def test_cache_hits_and_savings_show_up_in_metrics():
    app, gw = cached_app()
    with TestClient(app) as c:
        chat(c, cache=True)
        settle(c, gw)
        chat(c, cache=True)
        text = c.get("/metrics").text
    assert 'gw_cache_requests_total{layer="exact",result="hit"} 1.0' in text


# --- admin key ----------------------------------------------------------------------------------


def test_admin_key_guards_stats_metrics_and_admin_routes():
    with TestClient(make_app(admin_key_sha256=ADMIN_HASH)) as c:
        for path in ("/stats", "/metrics"):
            assert c.get(path).status_code == 401
            assert c.get(path, headers=AUTH).status_code == 401  # a client key is not an admin key
            assert c.get(path, headers=ADMIN).status_code == 200
        wrong = c.get("/stats", headers={"Authorization": "Bearer nope"})
        assert wrong.json()["error"]["code"] == "invalid_admin_key"


def test_admin_can_read_any_clients_budget_and_a_client_only_its_own():
    with TestClient(make_app(admin_key_sha256=ADMIN_HASH)) as c:
        mine = c.get("/admin/budget", headers=AUTH).json()
        theirs = c.get("/admin/budget?client=demo", headers=ADMIN).json()
        assert c.get("/admin/budget", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert mine["client"] == "full" and theirs["client"] == "demo"


def test_admin_invalidates_across_clients():
    with TestClient(make_app(cache_backend="memory", admin_key_sha256=ADMIN_HASH)) as c:
        r = c.post("/admin/cache/invalidate", headers=ADMIN, json={"tags": ["t"]})
    assert r.status_code == 200 and r.json()["tags"] == ["t"]


def test_without_an_admin_key_stats_are_open_in_dev_and_closed_in_prod():
    with TestClient(make_app("dev")) as c:
        assert c.get("/stats").status_code == 200
    with TestClient(make_app("prod")) as c:
        r = c.get("/metrics")
        assert r.status_code == 403 and r.json()["error"]["code"] == "admin_disabled"


# --- rate limit ---------------------------------------------------------------------------------


def clients_file(tmp_path, rpm):
    path = tmp_path / "clients.yaml"
    aliases = list(yaml.safe_load(open("config/routes.yaml", encoding="utf-8"))["aliases"])
    path.write_text(
        yaml.safe_dump(
            {
                "clients": {
                    "slow": {
                        "key_sha256": hashlib.sha256(b"k").hexdigest(),
                        "rpm": rpm,
                        "allowed_aliases": aliases,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_over_rpm_gets_429_with_retry_after(tmp_path):
    settings = Settings(
        _env_file=None,
        env="test",
        clients_file=clients_file(tmp_path, 2),
        providers_file=PROVIDERS_FILE,
        cache_backend="off",
        embedder="off",
        groq_api_key=None,
        google_api_key=None,
        anthropic_api_key=None,
    )
    with TestClient(create_app(settings)) as c:
        h = {"Authorization": "Bearer k"}
        codes = [c.post("/v1/chat/completions", headers=h, json=BODY).status_code for _ in range(3)]
        r = c.post("/v1/chat/completions", headers=h, json=BODY)
        metrics = c.get("/metrics").text
    assert codes == [200, 200, 429]
    assert r.json()["error"]["code"] == "rate_limit_exceeded" and int(r.headers["retry-after"]) >= 1
    assert 'gw_rate_limited_total{client="slow"}' in metrics


async def test_sliding_window_frees_slots_as_time_passes():
    now = [0.0]
    rl = RateLimiter(clock=lambda: now[0])
    assert [(await rl.check("a", 2))[0] for _ in range(3)] == [True, True, False]
    assert (await rl.check("b", 2))[0] is True  # other clients are unaffected
    now[0] = 61.0
    assert (await rl.check("a", 2))[0] is True


async def test_redis_limiter_shares_the_count_and_falls_back_when_redis_dies():
    redis = fakeredis.FakeAsyncRedis()
    rl = RateLimiter(redis=redis)
    assert [(await rl.check("a", 2))[0] for _ in range(3)] == [True, True, False]

    class Down:
        def register_script(self, _):
            async def boom(**_kw):
                from redis.exceptions import ConnectionError as RedisConnectionError

                raise RedisConnectionError("down")

            return boom

    rl = RateLimiter(redis=Down())
    assert [(await rl.check("a", 1))[0] for _ in range(2)] == [True, False]  # in-memory window took over
    assert rl.fallbacks == 2


# --- size and shape guardrail -------------------------------------------------------------------


def test_too_many_messages_and_images_are_rejected_before_any_call():
    app = make_app()
    app.state.gateway.routes.aliases["fast"].max_messages = 3
    app.state.gateway.routes.aliases["fast"].max_images = 1
    with TestClient(app) as c:
        many = [{"role": "user", "content": "x"}] * 4
        r = chat(c, messages=many)
        assert r.status_code == 400 and r.json()["error"]["code"] == "too_many_messages"
        img = {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}
        r = chat(c, messages=[{"role": "user", "content": [img, img]}])
        assert r.status_code == 400 and r.json()["error"]["code"] == "too_many_images"
        assert 'gw_guardrail_events_total{action="enforce",type="size"} 2.0' in c.get("/metrics").text


def test_oversized_body_is_413_in_openai_format():
    with TestClient(make_app(max_body_bytes=200)) as c:
        r = chat(c, messages=[{"role": "user", "content": "x" * 500}])
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


# --- access log (R16) ---------------------------------------------------------------------------


def records(caplog):
    return [json.loads(r.message) for r in caplog.records if r.name == "gateway.access"]


def test_one_json_line_per_request_and_never_the_prompt(caplog):
    caplog.set_level(logging.INFO, logger="gateway.access")
    with TestClient(make_app()) as c:
        chat(c, messages=[{"role": "user", "content": "my-secret-question"}])
        chat(c, model="nope")
    assert "my-secret-question" not in caplog.text and "Mock reply" not in caplog.text
    ok, bad = records(caplog)
    assert (
        ok["status"] == 200 and ok["client"] == "full" and ok["alias"] == "fast" and ok["provider"] == "mock"
    )
    assert ok["tokens_in"] and ok["tokens_out"] and ok["cost_usd"] == 0 and ok["attempts"] == 1
    assert bad["status"] == 404 and bad["error"] == "model_not_found" and bad["alias"] == "unknown"


def test_stream_logs_when_it_ends_with_its_outcome(caplog):
    caplog.set_level(logging.INFO, logger="gateway.access")
    with TestClient(make_app()) as c:
        chat(c, stream=True)
        text = c.get("/metrics").text
    (rec,) = records(caplog)
    assert rec["stream"] is True and rec["stream_outcome"] == "completed" and rec["tokens_out"]
    assert 'gw_stream_outcomes_total{outcome="completed"} 1.0' in text


def test_cache_hit_is_logged_as_a_hit_without_tokens(caplog):
    caplog.set_level(logging.INFO, logger="gateway.access")
    app, gw = cached_app()
    with TestClient(app) as c:
        chat(c, cache=True)
        settle(c, gw)
        chat(c, cache=True)
    first, second = records(caplog)
    assert first["cache"] is None and second["cache"] == "exact" and second["tokens_in"] is None


# --- R22: prompts in logs need Presidio ---------------------------------------------------------


def test_log_prompts_without_presidio_refuses_to_start(monkeypatch):
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)  # None makes the import fail
    with pytest.raises(PromptLoggingUnavailable, match="Presidio"):
        make_app(log_prompts=True)


def test_log_prompts_with_presidio_present_starts(monkeypatch):
    for name in ("presidio_analyzer", "presidio_anonymizer"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    assert make_app(log_prompts=True) is not None


def test_prompts_are_not_logged_by_default():
    assert Settings(_env_file=None).log_prompts is False


# --- tracing ------------------------------------------------------------------------------------


def traced_app():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = Settings(
        _env_file=None,
        env="test",
        clients_file=CLIENTS_FILE,
        providers_file=PROVIDERS_FILE,
        cache_backend="off",
        embedder="off",
        groq_api_key=None,
        google_api_key=None,
        anthropic_api_key=None,
    )
    return create_app(settings, tracer_provider=provider), exporter


def test_request_span_with_a_child_span_per_attempt():
    app, exporter = traced_app()
    with TestClient(app) as c:
        chat(c)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    req, attempt = spans["gateway.request"], spans["provider.attempt"]
    assert attempt.parent.span_id == req.context.span_id and attempt.context.trace_id == req.context.trace_id
    assert req.attributes["gen_ai.request.model"] == "fast" and req.attributes["gen_ai.system"] == "mock"
    assert req.attributes["gen_ai.usage.output_tokens"] > 0
    assert attempt.attributes["gen_ai.system"] == "mock"
    assert not any("hello there" in str(v) for s in spans.values() for v in s.attributes.values())


def test_traceparent_continues_the_callers_trace():
    app, exporter = traced_app()
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    with TestClient(app) as c:
        c.post("/v1/chat/completions", headers={**AUTH, "traceparent": tp}, json=BODY)
    req = next(s for s in exporter.get_finished_spans() if s.name == "gateway.request")
    assert format(req.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert format(req.parent.span_id, "016x") == "b7ad6b7169203331"


def test_no_tracing_without_an_endpoint():
    assert make_app().state.gateway.tracer.enabled is False
