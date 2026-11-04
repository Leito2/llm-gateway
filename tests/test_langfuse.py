"""Stretch · Langfuse export, against a mock HTTP server (never a real Langfuse)."""

import base64
import json

import httpx
from fastapi.testclient import TestClient
from helpers import FULL_KEY, make_app

from llm_gateway.telemetry.langfuse import LangfuseExporter

LINE = {
    "ts": 1_700_000_000.0, "request_id": "req_1", "client": "me", "alias": "fast", "provider": "groq",
    "model": "llama", "attempts": 1, "cache": None, "status": 200, "error": None, "stream": False,
    "stream_outcome": None, "ttft_ms": 120.0, "duration_ms": 250.0, "tokens_in": 10, "tokens_out": 20,
    "cost_usd": 0.0, "trace_id": "trace_1", "prompt_redacted": "SHOULD-NEVER-LEAVE",
}  # fmt: skip


def exporter(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LangfuseExporter("https://lf.test/", "pk-lf-1", "sk-lf-1", client, **kw)


async def test_a_request_becomes_a_trace_and_a_generation_without_prompt_text():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(207, json={"successes": [], "errors": []})

    ex = exporter(handler)
    ex.record(LINE)
    await ex.flush()
    (req,) = seen
    assert str(req.url) == "https://lf.test/api/public/ingestion"
    assert req.headers["authorization"] == "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-1").decode()
    raw = req.content.decode()
    assert "SHOULD-NEVER-LEAVE" not in raw
    trace, gen = json.loads(raw)["batch"]
    assert trace["type"] == "trace-create" and trace["body"]["id"] == "trace_1"
    g = gen["body"]
    assert gen["type"] == "generation-create" and g["traceId"] == "trace_1" and g["model"] == "llama"
    assert g["usage"] == {"input": 10, "output": 20, "unit": "TOKENS", "totalCost": 0.0}
    assert g["completionStartTime"] > g["startTime"] and g["level"] == "DEFAULT"
    assert ex.sent == 2 and ex.errors == 0


async def test_errors_are_marked_and_a_dead_langfuse_is_counted_not_raised():
    ex = exporter(lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")))
    ex.record({**LINE, "status": 503, "error": "all_providers_unavailable"})
    await ex.flush()
    assert ex.errors == 1 and ex.sent == 0


async def test_a_rejection_is_counted():
    ex = exporter(lambda r: httpx.Response(401))
    ex.record(LINE)
    await ex.flush()
    assert ex.errors == 1 and ex.sent == 0


async def test_a_full_queue_drops_events_instead_of_blocking():
    ex = exporter(lambda r: httpx.Response(207), max_queue=3)
    ex.record(LINE)  # 2 events
    ex.record(LINE)  # the 4th event does not fit
    assert ex.dropped == 1 and ex.snapshot()["queued"] == 3


async def test_batches_are_split_by_size():
    sizes = []

    def handler(request):
        sizes.append(len(json.loads(request.content)["batch"]))
        return httpx.Response(207)

    ex = exporter(handler, batch_size=3)
    for _ in range(4):
        ex.record(LINE)  # 8 events
    await ex.flush()
    assert sizes == [3, 3, 2]


def test_the_app_exports_each_request_and_reports_it_in_stats():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["batch"])
        return httpx.Response(207)

    app = make_app(langfuse_host="https://lf.test", langfuse_public_key="pk", langfuse_secret_key="sk")
    gw = app.state.gateway
    gw.langfuse._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(app, headers={"Authorization": f"Bearer {FULL_KEY}"}) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "fast", "messages": [{"role": "user", "content": "secret-q"}]},
        )
        assert r.status_code == 200
    assert sent, "the exporter flushes on shutdown"
    events = [e for batch in sent for e in batch]
    assert {e["type"] for e in events} == {"trace-create", "generation-create"}
    assert "secret-q" not in json.dumps(events) and "Mock reply" not in json.dumps(events)


def test_without_all_three_settings_there_is_no_exporter():
    assert make_app(langfuse_host="https://lf.test").state.gateway.langfuse is None
