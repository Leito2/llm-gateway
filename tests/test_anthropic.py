"""M7: the Anthropic adapter against fixtures written from the documented Messages API (ADR-0005). $0, offline."""

import json

import httpx
import pytest

from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.providers.anthropic import (
    AnthropicProvider,
    classify,
    normalize_usage,
    split_messages,
)
from llm_gateway.providers.base import ProviderError


def request(**kw) -> ChatCompletionRequest:
    base = {"model": "smart_paid", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    return ChatCompletionRequest.model_validate({**base, **kw})


def provider(handler) -> tuple[AnthropicProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return AnthropicProvider("sk-test", client, "https://anthropic.test"), seen


MESSAGE = {
    "id": "msg_01ABCDEFGHIJKLMNOPQRSTUV",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "thinking", "thinking": "let me think", "signature": "sig"},
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "there"},
    ],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 20,
        "cache_read_input_tokens": 600,
        "cache_creation_input_tokens": 100,
        "output_tokens": 7,
    },
}


def sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


STREAM = sse(
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_01STREAMSTREAMSTREAMSTRE",
                "model": "claude-haiku-5-5",
                "usage": {"input_tokens": 25, "cache_read_input_tokens": 0, "output_tokens": 1},
            },
        },
    ),
    (
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
    ),
    ("ping", {"type": "ping"}),
    ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hel"}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "lo"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    (
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12}},
    ),
    ("message_stop", {"type": "message_stop"}),
)


# --- pure translation ----------------------------------------------------------------------------


def test_usage_follows_the_openai_convention():
    u = normalize_usage(MESSAGE["usage"])
    assert u["prompt_tokens"] == 720 and u["completion_tokens"] == 7 and u["total_tokens"] == 727
    assert u["prompt_tokens_details"] == {"cached_tokens": 600, "cache_creation_tokens": 100}
    assert "prompt_tokens_details" not in normalize_usage({"input_tokens": 5, "output_tokens": 1})


def test_system_goes_to_the_system_field_and_turns_alternate():
    req = request(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
            {"role": "tool", "content": "d"},
        ]
    )
    system, turns = split_messages(req.messages, prompt_cache=False)
    assert system == [{"type": "text", "text": "Be brief."}]
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert [b["text"] for b in turns[0]["content"]] == ["a", "b"]


def test_cache_control_only_when_enabled_and_the_prompt_reaches_the_minimum():
    big = "x" * 3000  # ~1000 estimated tokens
    msgs = request(messages=[{"role": "system", "content": big}, {"role": "user", "content": "q"}]).messages
    assert split_messages(msgs, prompt_cache=True)[0][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in split_messages(msgs, prompt_cache=False)[0][-1]
    small = request(
        messages=[{"role": "system", "content": "short"}, {"role": "user", "content": "q"}]
    ).messages
    assert "cache_control" not in split_messages(small, prompt_cache=True)[0][-1]


def test_images_become_image_blocks():
    parts = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.jpg"}},
    ]
    _, turns = split_messages(request(messages=[{"role": "user", "content": parts}]).messages, False)
    blocks = turns[0]["content"]
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert blocks[2]["source"] == {"type": "url", "url": "https://x.test/a.jpg"}


@pytest.mark.parametrize(
    "status, message, etype, kind",
    [
        (400, "messages: bad", "invalid_request_error", "client"),
        (400, "prompt is too long: 250000 tokens", "invalid_request_error", "provider"),
        (401, "", "authentication_error", "provider"),
        (404, "", "not_found_error", "provider"),
        (413, "", "request_too_large", "client"),
        (429, "", "rate_limit_error", "transient"),
        (500, "", "api_error", "transient"),
        (529, "", "overloaded_error", "transient"),
    ],
)
def test_error_kinds(status, message, etype, kind):
    assert classify(status, message, etype) == kind


# --- non-streaming -------------------------------------------------------------------------------


async def test_complete_translates_the_response_and_sends_the_right_request():
    p, seen = provider(lambda r: httpx.Response(200, json=MESSAGE))
    out = await p.complete(
        request(temperature=0.7, reasoning_effort="low", prompt_cache=True), "claude-haiku-5-5"
    )
    assert out["choices"][0]["message"]["content"] == "Hello there"  # thinking block is not forwarded
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == 720
    assert out["_meta"] == {"params_dropped": ["temperature"], "effort": "low"}
    sent = seen[0]
    assert sent.url == "https://anthropic.test/v1/messages"
    assert sent.headers["x-api-key"] == "sk-test" and sent.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(sent.content)
    assert (
        "temperature" not in body and body["max_tokens"] == 100 and body["output_config"] == {"effort": "low"}
    )
    assert body["model"] == "claude-haiku-5-5" and body["stream"] is False


async def test_default_temperature_is_not_reported_as_dropped():
    p, _ = provider(lambda r: httpx.Response(200, json=MESSAGE))
    out = await p.complete(request(), "claude-haiku-5-5")  # the client never set temperature
    assert "_meta" not in out


@pytest.mark.parametrize(
    "stop, finish",
    [
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_calls"),
    ],
)
async def test_stop_reasons_map_to_finish_reasons(stop, finish):
    p, _ = provider(lambda r: httpx.Response(200, json={**MESSAGE, "stop_reason": stop}))
    out = await p.complete(request(), "claude-haiku-5-5")
    assert out["choices"][0]["finish_reason"] == finish


async def test_json_format_is_asked_for_in_the_system_prompt():
    p, seen = provider(lambda r: httpx.Response(200, json=MESSAGE))
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    await p.complete(request(response_format=fmt), "claude-haiku-5-5")
    assert "JSON Schema" in json.loads(seen[0].content)["system"][-1]["text"]


async def test_http_errors_become_typed_provider_errors():
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    p, _ = provider(lambda r: httpx.Response(429, json=body, headers={"retry-after": "7"}))
    with pytest.raises(ProviderError) as exc:
        await p.complete(request(), "claude-haiku-5-5")
    assert exc.value.kind == "transient" and exc.value.status == 429 and exc.value.retry_after == 7.0


async def test_a_client_error_carries_the_upstream_message():
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: too big"}}
    p, _ = provider(lambda r: httpx.Response(400, json=body))
    with pytest.raises(ProviderError) as exc:
        await p.complete(request(), "claude-haiku-5-5")
    assert exc.value.kind == "client" and "max_tokens: too big" in str(exc.value)


async def test_network_failure_is_transient():
    def boom(_r):
        raise httpx.ConnectError("down")

    p, _ = provider(boom)
    with pytest.raises(ProviderError) as exc:
        await p.complete(request(), "claude-haiku-5-5")
    assert exc.value.kind == "transient"


# --- streaming -----------------------------------------------------------------------------------


async def collect(p, req):
    return [c async for c in p.stream(req, "claude-haiku-5-5")]


async def test_stream_translates_events_to_chunks():
    p, _ = provider(lambda r: httpx.Response(200, text=STREAM, headers={"content-type": "text/event-stream"}))
    chunks = await collect(p, request(temperature=1.0))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[0]["_meta"] == {"params_dropped": ["temperature"]}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hello"
    # thinking deltas and pings are chunks without choices: liveness for the stall timer, nothing to forward
    assert sum(1 for c in chunks if not c["choices"]) == 2
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"] == {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37}
    assert all(c["id"] == chunks[0]["id"] for c in chunks)


async def test_stream_refusal_is_content_filter():
    events = STREAM.replace('"end_turn"', '"refusal"')
    p, _ = provider(lambda r: httpx.Response(200, text=events))
    assert (await collect(p, request()))[-1]["choices"][0]["finish_reason"] == "content_filter"


async def test_stream_error_event_is_typed():
    events = sse(
        ("message_start", {"type": "message_start", "message": {"id": "m", "usage": {}}}),
        ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
    )
    p, _ = provider(lambda r: httpx.Response(200, text=events))
    got = []
    with pytest.raises(ProviderError) as exc:
        async for c in p.stream(request(), "claude-haiku-5-5"):
            got.append(c)
    assert exc.value.kind == "transient" and len(got) == 1


async def test_stream_cut_before_message_stop_is_an_error():
    cut = STREAM.split("event: message_delta")[0]
    p, _ = provider(lambda r: httpx.Response(200, text=cut))
    with pytest.raises(ProviderError) as exc:
        await collect(p, request())
    assert exc.value.kind == "transient"


async def test_stream_http_error_before_any_event():
    body = {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
    p, _ = provider(lambda r: httpx.Response(401, json=body))
    with pytest.raises(ProviderError) as exc:
        await collect(p, request())
    assert exc.value.kind == "provider" and exc.value.status == 401


# --- through the gateway -------------------------------------------------------------------------


def test_effort_is_clamped_to_the_alias_ceiling():
    from llm_gateway.api.chat import clamp_effort

    assert clamp_effort("high", "low") == "low"
    assert clamp_effort("low", "high") == "low"
    assert clamp_effort(None, "low") == "low"
    assert clamp_effort("high", None) == "high"
    assert clamp_effort(None, None) is None


def test_paid_alias_end_to_end_headers_and_cost():
    from fastapi.testclient import TestClient
    from helpers import FULL_KEY, make_app

    app = make_app(run_budget_usd=1.0)
    gw = app.state.gateway
    p, seen = provider(lambda r: httpx.Response(200, json=MESSAGE))
    gw.providers["anthropic"] = p
    auth = {"Authorization": f"Bearer {FULL_KEY}"}
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            headers=auth,
            json={
                "model": "smart_paid",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
                "reasoning_effort": "high",
            },
        )
        budget = c.get("/admin/budget", headers=auth).json()
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "anthropic:claude-haiku-5-5"
    assert r.headers["x-params-dropped"] == "temperature" and r.headers["x-effort"] == "low"
    assert "_meta" not in r.json()
    assert json.loads(seen[0].content)["output_config"] == {"effort": "low"}  # clamped from "high"
    # plain 20*0.10 + read 600*0.01 + write 100*0.125 + out 7*0.50 = 24 micro-dollars
    assert budget["scopes"]["run"]["spent"] == 24 and budget["scopes"]["run"]["reserved"] == 0


def test_paid_alias_is_not_registered_without_a_run_budget():
    from helpers import make_app

    gw = make_app(anthropic_api_key="sk-x", run_budget_usd=0.0).state.gateway
    assert "anthropic" not in gw.providers
