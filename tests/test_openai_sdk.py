"""Contract test (G7, partial): the official OpenAI SDK works against the gateway unchanged."""
import openai
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from llm_gateway.main import create_app
from llm_gateway.settings import Settings


@pytest.fixture
def sdk() -> OpenAI:
    http = TestClient(create_app(Settings(env="test")))
    return OpenAI(base_url="http://testserver/v1", api_key="test-key", http_client=http, max_retries=0)


def test_chat_completion(sdk):
    r = sdk.chat.completions.create(model="fast", messages=[{"role": "user", "content": "hola"}])
    assert r.choices[0].message.content == "Mock reply to: hola"
    assert r.choices[0].finish_reason == "stop" and r.usage.total_tokens > 0


def test_streaming_with_usage(sdk):
    stream = sdk.chat.completions.create(model="smart", messages=[{"role": "user", "content": "uno dos"}],
                                         stream=True, stream_options={"include_usage": True})
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Mock reply to: uno dos"
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 5


def test_models_lists_aliases(sdk):
    ids = {m.id for m in sdk.models.list().data}
    assert {"fast", "smart", "judge", "smart_paid"} <= ids


def test_unknown_alias_raises_not_found(sdk):
    with pytest.raises(openai.NotFoundError) as exc:
        sdk.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert exc.value.code == "model_not_found"


def test_invalid_request_raises_bad_request(sdk):
    with pytest.raises(openai.BadRequestError):
        sdk.chat.completions.create(model="fast", messages=[])
