"""OpenAICompatProvider against a fake HTTP server (httpx.MockTransport): no network, no quota."""

import json

import httpx
import pytest

from llm_gateway.api.schemas import ChatCompletionRequest, ResponseFormat
from llm_gateway.providers.base import ProviderError
from llm_gateway.providers.openai_compat import OpenAICompatProvider

REQ = ChatCompletionRequest(model="fast", messages=[{"role": "user", "content": "hola mundo"}], max_tokens=50)
COMPLETION = {
    "id": "x",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def provider(handler, api_key="secret") -> OpenAICompatProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider("groq", "http://upstream/v1/", api_key, client)


def sse(*chunks: dict | str) -> bytes:
    return "".join(f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks).encode()


def delta(text: str, **extra) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        **extra,
    }


async def collect(p: OpenAICompatProvider) -> list[dict]:
    return [c async for c in p.stream(REQ, "llama")]


async def test_complete_sends_the_openai_request_and_returns_the_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=COMPLETION)

    assert await provider(handler).complete(REQ, "llama") == COMPLETION
    assert seen["url"] == "http://upstream/v1/chat/completions" and seen["auth"] == "Bearer secret"
    assert seen["body"] == {
        "model": "llama",
        "messages": [{"role": "user", "content": "hola mundo"}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 50,
    }


async def test_no_authorization_header_without_a_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        return httpx.Response(200, json=COMPLETION)

    await provider(handler, api_key=None).complete(REQ, "llama")
    assert "authorization" not in seen["headers"]


async def test_missing_usage_is_estimated_upwards():
    body = {**COMPLETION, "usage": None}
    data = await provider(lambda r: httpx.Response(200, json=body)).complete(REQ, "llama")
    assert data["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}  # ceil(len / 3)


async def test_response_format_is_passed_through():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=COMPLETION)

    req = REQ.model_copy(update={"response_format": ResponseFormat(type="json_object")})
    await provider(handler).complete(req, "llama")
    assert seen["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "status,body,kind",
    [
        (400, {"error": {"message": "bad param", "code": "invalid"}}, "client"),
        (400, {"error": {"message": "x", "code": "context_length_exceeded"}}, "provider"),
        (400, {"error": {"message": "This model's maximum context length is 8192 tokens"}}, "provider"),
        (401, {"error": {"message": "Invalid API Key"}}, "provider"),
        (404, {"error": {"message": "model not found"}}, "provider"),
        (429, {"error": {"message": "rate"}}, "transient"),
        (500, {}, "transient"),
        (503, {"error": {"message": "overloaded"}}, "transient"),
    ],
)
async def test_http_errors_are_classified(status, body, kind):
    with pytest.raises(ProviderError) as exc:
        await provider(lambda r: httpx.Response(status, json=body)).complete(REQ, "llama")
    assert exc.value.kind == kind and exc.value.status == status


async def test_only_client_errors_carry_the_upstream_message():
    def failing(status, message):
        return provider(lambda r: httpx.Response(status, json={"error": {"message": message}}))

    with pytest.raises(ProviderError) as client_err:
        await failing(400, "temperature too high").complete(REQ, "m")
    assert "temperature too high" in str(client_err.value)
    with pytest.raises(ProviderError) as other:
        await failing(401, "key sk-live-123 is bad").complete(REQ, "m")
    assert "sk-live" not in str(other.value) and "HTTP 401" in str(other.value)


async def test_429_keeps_retry_after():
    with pytest.raises(ProviderError) as exc:
        await provider(lambda r: httpx.Response(429, headers={"retry-after": "12"})).complete(REQ, "m")
    assert exc.value.retry_after == 12.0


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
async def test_network_failures_are_transient(error):
    def handler(request):
        raise error

    with pytest.raises(ProviderError) as exc:
        await provider(handler).complete(REQ, "llama")
    assert exc.value.kind == "transient"


async def test_malformed_success_body_is_a_provider_error():
    with pytest.raises(ProviderError) as exc:
        await provider(lambda r: httpx.Response(200, json={"nope": 1})).complete(REQ, "llama")
    assert exc.value.kind == "provider"


async def test_stream_yields_chunks_asks_for_usage_and_stops_at_done():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        final = {**delta(""), "choices": [], "usage": usage}
        body = sse(delta("ho"), delta("la"), final, "[DONE]", delta("never read"))
        return httpx.Response(200, content=body)

    chunks = await collect(provider(handler))
    assert seen["body"]["stream"] is True and seen["body"]["stream_options"] == {"include_usage": True}
    assert [c["choices"][0]["delta"]["content"] for c in chunks if c["choices"]] == ["ho", "la"]
    assert chunks[-1]["usage"]["total_tokens"] == 5 and len(chunks) == 3


async def test_stream_without_usage_gets_an_estimated_final_chunk():
    chunks = await collect(provider(lambda r: httpx.Response(200, content=sse(delta("abcdef"), "[DONE]"))))
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }


async def test_stream_http_error_is_raised_before_any_chunk():
    p = provider(lambda r: httpx.Response(429, headers={"retry-after": "5"}))
    with pytest.raises(ProviderError) as exc:
        await collect(p)
    assert exc.value.kind == "transient" and exc.value.retry_after == 5.0


async def test_empty_stream_is_a_provider_error():
    with pytest.raises(ProviderError, match="empty"):
        await collect(provider(lambda r: httpx.Response(200, content=sse("[DONE]"))))


async def test_error_object_inside_the_stream_is_a_provider_error():
    with pytest.raises(ProviderError):
        await collect(provider(lambda r: httpx.Response(200, content=sse({"error": {"message": "boom"}}))))


async def test_connection_failure_while_streaming_is_transient():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError) as exc:
        await collect(provider(handler))
    assert exc.value.kind == "transient"
