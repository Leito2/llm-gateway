"""The interface every provider adapter implements (implementations arrive in M1+)."""
from collections.abc import AsyncIterator
from typing import Protocol

from llm_gateway.api.schemas import ChatCompletionRequest


class ProviderError(Exception):
    """Raised by adapters; `retryable` tells the resilience layer whether to retry or fall back."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class Provider(Protocol):
    name: str

    async def complete(self, request: ChatCompletionRequest, model: str) -> dict:
        """Non-streaming completion in OpenAI `chat.completion` shape, including `usage`."""
        ...

    def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[dict]:
        """Streaming completion yielding OpenAI `chat.completion.chunk` dicts; last chunk carries `usage`."""
        ...
