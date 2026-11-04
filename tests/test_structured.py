"""M7: structured output — validation, repair, fallback, buffered streaming (PLAN §3.2, R15)."""

import json

import pytest
from fastapi.testclient import TestClient
from helpers import FULL_KEY, make_app

from llm_gateway.api.schemas import ChatCompletionRequest, ResponseFormat
from llm_gateway.providers.mock import MockProvider
from llm_gateway.structured.validate import (
    InvalidSchema,
    build_validator,
    merge_usage,
    repair_request,
)

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
FMT = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA}}
GOOD = '{"name": "Ada", "age": 36}'
HEADERS = {"Authorization": f"Bearer {FULL_KEY}"}


def completion(content: str, finish: str = "stop", usage=None) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# --- validator ----------------------------------------------------------------------------------


def validator():
    return build_validator(ResponseFormat.model_validate(FMT))


def test_text_format_needs_no_validator():
    assert build_validator(None) is None
    assert build_validator(ResponseFormat(type="text")) is None


def test_valid_reply_passes_untouched():
    c = completion(GOOD)
    assert validator().check(c) is None and "_meta" not in c


def test_code_fence_and_surrounding_prose_are_repaired_without_a_new_call():
    for wrapped in (f"```json\n{GOOD}\n```", f"Sure! Here it is: {GOOD} Hope that helps."):
        c = completion(wrapped)
        assert validator().check(c) is None
        assert c["choices"][0]["message"]["content"] == GOOD
        assert c["_meta"]["structured_repaired"] is True


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("not json at all", "not valid JSON"),
        ("", "empty"),
        ('{"name": "Ada"}', "age"),
        ('{"name": "Ada", "age": "x"}', "/age" if False else "age"),
        ('{"name": "Ada", "age": 1, "extra": 1}', "extra"),
        ("42", "must be a JSON object"),
    ],
)
def test_invalid_replies_say_what_is_wrong(reply, expected):
    problem = validator().check(completion(reply))
    assert problem and expected in problem


def test_truncated_reply_is_reported_as_cut_off():
    assert "cut off" in validator().check(completion('{"name": "Ada", "a', finish="length"))


def test_json_object_rejects_arrays():
    v = build_validator(ResponseFormat(type="json_object"))
    assert v.check(completion('{"a": 1}')) is None
    assert v.check(completion("[1, 2]")) is not None


def test_a_broken_schema_is_rejected_up_front():
    with pytest.raises(InvalidSchema):
        build_validator(ResponseFormat(type="json_schema", json_schema={"schema": {"type": "nope"}}))
    with pytest.raises(InvalidSchema):
        build_validator(ResponseFormat(type="json_schema", json_schema={"name": "x"}))


def test_repair_request_carries_the_bad_reply_and_the_problem():
    req = ChatCompletionRequest(model="fast", messages=[{"role": "user", "content": "who?"}])
    out = repair_request(req, completion("oops"), "the reply is not valid JSON")
    assert [m.role for m in out.messages] == ["user", "assistant", "user"]
    assert out.messages[1].content == "oops" and "not valid JSON" in out.messages[2].content


def test_merge_usage_sums_calls_and_cache_details():
    a = {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}}
    b = {"prompt_tokens": 20, "completion_tokens": 3, "prompt_tokens_details": {"cache_creation_tokens": 5}}
    assert merge_usage(a, b) == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
        "prompt_tokens_details": {"cached_tokens": 4, "cache_creation_tokens": 5},
    }
    assert merge_usage(None, b) == b


# --- through the gateway ------------------------------------------------------------------------


class Scripted(MockProvider):
    """Replies with the scripted strings in order (the last one repeats) and counts calls."""

    def __init__(self, *replies: str) -> None:
        super().__init__()
        self.replies, self.calls, self.stream_calls, self.seen = list(replies), 0, 0, []

    async def complete(self, request, model):
        self.seen.append(request)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return completion(reply)

    async def stream(self, request, model):
        self.stream_calls += 1
        async for chunk in super().stream(request, model):
            yield chunk


def gateway(groq: Scripted, mock: Scripted | None = None):
    app = make_app()
    gw = app.state.gateway
    gw.providers["groq"] = groq
    gw.providers["mock"] = mock or Scripted(GOOD)
    return TestClient(app), gw


def ask(client, **extra):
    body = {"model": "fast", "messages": [{"role": "user", "content": "who?"}], "response_format": FMT}
    return client.post("/v1/chat/completions", headers=HEADERS, json={**body, **extra})


def test_valid_reply_is_returned_as_is():
    groq = Scripted(GOOD)
    client, _ = gateway(groq)
    r = ask(client)
    assert r.status_code == 200 and groq.calls == 1
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"name": "Ada", "age": 36}
    assert "x-structured-repaired" not in r.headers


def test_fenced_reply_is_repaired_and_flagged():
    client, _ = gateway(Scripted(f"```json\n{GOOD}\n```"))
    r = ask(client)
    assert r.status_code == 200 and r.headers["x-structured-repaired"] == "1"
    assert r.json()["choices"][0]["message"]["content"] == GOOD
    assert "_meta" not in r.json()


def test_one_retry_on_the_same_provider_with_the_problem_as_context():
    groq = Scripted("not json", GOOD)
    client, _ = gateway(groq)
    r = ask(client)
    assert r.status_code == 200 and groq.calls == 2
    assert (
        r.headers["x-gateway-provider"].startswith("groq:") and r.headers["x-gateway-fallback-depth"] == "1"
    )
    retry = groq.seen[1].messages
    assert retry[-2].content == "not json" and "rejected" in retry[-1].content
    assert r.json()["usage"]["prompt_tokens"] == 20  # both calls are counted


def test_still_invalid_after_the_retry_falls_back_to_the_next_provider():
    groq, mock = Scripted("nope"), Scripted(GOOD)
    client, gw = gateway(groq, mock)
    r = ask(client)
    assert r.status_code == 200 and (groq.calls, mock.calls) == (2, 1)
    assert r.headers["x-gateway-provider"].startswith("mock:")
    assert gw.breakers.get("groq").snapshot()["consecutive_failures"] == 0  # quality, not availability


def test_every_provider_invalid_is_a_502_within_the_attempt_limit():
    groq, mock = Scripted("nope"), Scripted("also nope")
    client, _ = gateway(groq, mock)
    r = ask(client)
    assert r.status_code == 502 and r.json()["error"]["code"] == "invalid_structured_output"
    assert groq.calls + mock.calls <= 5


def test_streaming_is_buffered_validated_and_replayed():
    groq = Scripted(f"```json\n{GOOD}\n```")
    client, _ = gateway(groq)
    r = ask(client, stream=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    text = "".join(
        c["delta"].get("content", "") for e in events[:-1] for c in json.loads(e).get("choices", [])
    )
    assert text == GOOD and groq.stream_calls == 0 and groq.calls == 1


def test_streaming_with_unfixable_output_is_a_plain_502_not_a_broken_stream():
    client, _ = gateway(Scripted("nope"), Scripted("nope"))
    r = ask(client, stream=True)
    assert r.status_code == 502 and r.json()["error"]["code"] == "invalid_structured_output"


def test_invalid_schema_is_a_400_before_any_provider_call():
    groq = Scripted(GOOD)
    client, _ = gateway(groq)
    r = ask(client, response_format={"type": "json_schema", "json_schema": {"schema": {"type": "nope"}}})
    assert r.status_code == 400 and r.json()["error"]["param"] == "response_format" and groq.calls == 0


def test_plain_text_requests_are_not_validated():
    groq = Scripted("just words")
    client, _ = gateway(groq)
    r = ask(client, response_format={"type": "text"})
    assert r.status_code == 200 and groq.calls == 1
