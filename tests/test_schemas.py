import pytest
from pydantic import ValidationError

from llm_gateway.api.schemas import ChatCompletionRequest


def test_minimal_request_is_valid():
    req = ChatCompletionRequest(model="fast", messages=[{"role": "user", "content": "hola"}])
    assert req.temperature == 0.0 and req.stream is False


def test_empty_messages_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="fast", messages=[])
