import json

from fastapi.testclient import TestClient
from helpers import make_app, make_client

from llm_gateway.providers.mock import MockProvider

MSG = [{"role": "user", "content": "hola mundo"}]


def sse_data(text: str) -> list[str]:
    return [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]


def test_req_01_response_shape_and_headers():
    r = make_client().post("/v1/chat/completions", json={"model": "fast", "messages": MSG})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Mock reply to: hola mundo"}
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert r.headers["x-alias"] == "fast"
    assert r.headers["x-gateway-provider"] == "mock:mock-fast"
    assert r.headers["x-gateway-fallback-depth"] == "1"
    assert r.headers["x-request-id"]


def test_client_request_id_is_echoed_and_unsafe_ones_replaced():
    c = make_client()
    body = {"model": "fast", "messages": MSG}
    ok = c.post("/v1/chat/completions", json=body, headers={"X-Request-ID": "abc-123"})
    assert ok.headers["x-request-id"] == "abc-123"
    bad = c.post("/v1/chat/completions", json=body, headers={"X-Request-ID": "a b<c>"})
    assert bad.headers["x-request-id"] != "a b<c>"


def test_req_17_unknown_alias_is_404_with_valid_aliases():
    r = make_client().post("/v1/chat/completions", json={"model": "gpt-4o", "messages": MSG})
    assert r.status_code == 404
    err = r.json()["error"]
    assert (err["code"], err["type"], err["param"]) == ("model_not_found", "invalid_request_error", "model")
    assert "fast" in err["message"] and "smart" in err["message"]
    assert r.headers["x-request-id"]


def test_req_19_max_tokens_is_clamped_to_the_alias_cap():
    c = make_client()
    r = c.post("/v1/chat/completions", json={"model": "fast", "messages": MSG, "max_tokens": 4000})
    assert r.status_code == 200 and r.headers["x-max-tokens-clamped"] == "256"
    ok = c.post("/v1/chat/completions", json={"model": "fast", "messages": MSG, "max_tokens": 10})
    assert "x-max-tokens-clamped" not in ok.headers


def test_max_tokens_reaches_the_provider():
    long_text = " ".join(["palabra"] * 600)
    r = make_client().post("/v1/chat/completions", json={"model": "fast", "messages": [
        {"role": "user", "content": long_text}], "max_tokens": 4000})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert r.json()["usage"]["completion_tokens"] == 256


def test_invalid_request_is_400_in_openai_format():
    r = make_client().post("/v1/chat/completions", json={"model": "fast", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request" and r.json()["error"]["param"] == "messages"


def test_degraded_only_chain_is_503_when_the_alias_forbids_it():
    r = make_client().post("/v1/chat/completions", json={"model": "judge_golden", "messages": MSG})
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_providers_unavailable"


def test_mock_is_ignored_outside_dev_and_test():
    r = make_client("prod").post("/v1/chat/completions", json={"model": "fast", "messages": MSG})
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_providers_unavailable"
    assert "GW_ENV=prod" in r.json()["error"]["message"]


def test_stream_is_sse_with_done_and_matches_the_non_stream_answer():
    c = make_client()
    plain = c.post("/v1/chat/completions", json={"model": "fast", "messages": MSG}).json()
    r = c.post("/v1/chat/completions", json={"model": "fast", "messages": MSG, "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"
    assert r.headers["x-gateway-provider"] == "mock:mock-fast" and r.headers["x-alias"] == "fast"
    events = sse_data(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(ch["object"] == "chat.completion.chunk" for ch in chunks)
    assert "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks) == \
        plain["choices"][0]["message"]["content"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" not in ch for ch in chunks)   # usage only when the client asks for it


def test_stream_usage_chunk_only_with_include_usage():
    r = make_client().post("/v1/chat/completions", json={
        "model": "fast", "messages": MSG, "stream": True, "stream_options": {"include_usage": True}})
    chunks = [json.loads(e) for e in sse_data(r.text)[:-1]]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0
    assert all("usage" not in ch for ch in chunks[:-1])
    assert len({ch["id"] for ch in chunks}) == 1


def test_stream_closes_the_provider_stream():
    """The upstream generator must be closed when the response ends (same path as a client disconnect)."""
    closed = []

    class Tracking(MockProvider):
        async def stream(self, request, model):
            try:
                async for chunk in super().stream(request, model):
                    yield chunk
            finally:
                closed.append(True)

    app = make_app()
    app.state.gateway.providers["mock"] = Tracking()
    client = TestClient(app, headers=make_client().headers)
    client.post("/v1/chat/completions", json={"model": "fast", "messages": MSG, "stream": True})
    assert closed == [True]


def test_stats_counts_requests_without_client_controlled_labels():
    c = make_client()
    c.post("/v1/chat/completions", json={"model": "fast", "messages": MSG})
    c.post("/v1/chat/completions", json={"model": "no-such-alias", "messages": MSG})
    stats = c.get("/stats").json()
    assert stats["requests_total"] == 2
    assert {(r["alias"], r["provider"], r["status"]) for r in stats["requests"]} == {
        ("fast", "mock", 200), ("unknown", "-", 404)}


def test_ui_is_served_with_a_restrictive_csp():
    r = make_client().get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "connect-src 'self'" in r.headers["content-security-policy"]
    assert "Authorization" in r.text and "sessionStorage" in r.text
