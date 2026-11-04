"""Deterministic provider for tests, demos and offline use. Only registered when GW_ENV is dev or test."""
import asyncio
import time
import uuid
from collections.abc import AsyncIterator

from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.pricing.tokens import content_text

_text = content_text


class MockProvider:
    """Replies "Mock reply to: <last user message>" one word per token, honouring `max_tokens`."""

    name = "mock"

    def __init__(self, *, ttft_s: float = 0.0, token_delay_s: float = 0.0) -> None:
        self.ttft_s = ttft_s
        self.token_delay_s = token_delay_s

    def _reply(self, request: ChatCompletionRequest) -> tuple[list[str], str, dict]:
        last_user = next((_text(m.content) for m in reversed(request.messages) if m.role == "user"), "")
        words = f"Mock reply to: {last_user}".split()
        finish = "stop"
        if request.max_tokens is not None and len(words) > request.max_tokens:
            words, finish = words[: request.max_tokens], "length"
        prompt = max(1, sum(len(_text(m.content).split()) for m in request.messages))
        done = len(words)
        usage = {"prompt_tokens": prompt, "completion_tokens": done, "total_tokens": prompt + done}
        return words, finish, usage

    async def complete(self, request: ChatCompletionRequest, model: str) -> dict:
        words, finish, usage = self._reply(request)
        await asyncio.sleep(self.ttft_s + self.token_delay_s * len(words))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(words)},
                         "finish_reason": finish}],
            "usage": usage,
        }

    async def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[dict]:
        words, finish, usage = self._reply(request)
        base = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model}

        def chunk(delta: dict, finish_reason: str | None = None, **extra) -> dict:
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            return {**base, "choices": [choice], **extra}

        await asyncio.sleep(self.ttft_s)
        yield chunk({"role": "assistant", "content": ""})
        for i, word in enumerate(words):
            await asyncio.sleep(self.token_delay_s)
            yield chunk({"content": word if i == 0 else " " + word})
        yield chunk({}, finish, usage=usage)   # Provider contract: the last chunk carries `usage`
