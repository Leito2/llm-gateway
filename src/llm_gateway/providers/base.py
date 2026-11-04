"""The interface every provider adapter implements."""
from collections.abc import AsyncIterator
from typing import Literal, Protocol

from llm_gateway.api.schemas import ChatCompletionRequest

# PLAN §3.1. transient: retry/fallback, counts for the breaker. provider: fallback, counts for the breaker.
# client: the request itself is invalid, so it goes back to the client and no other provider is tried.
ErrorKind = Literal["transient", "provider", "client"]


class ProviderError(Exception):
    """Raised by adapters; `kind` tells the resilience layer whether to retry, fall back or give up."""

    def __init__(self, message: str, *, kind: ErrorKind, status: int | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after     # seconds, from the provider's Retry-After header
        self.provider: str | None = None   # set by the resilience layer, which knows which link failed


class Provider(Protocol):
    name: str

    async def complete(self, request: ChatCompletionRequest, model: str) -> dict:
        """Non-streaming completion in OpenAI `chat.completion` shape, including `usage`."""
        ...

    def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[dict]:
        """Streaming completion yielding OpenAI `chat.completion.chunk` dicts; last chunk carries `usage`."""
        ...
