"""One adapter for every OpenAI-compatible endpoint: Ollama (/v1), Groq, Google AI Studio, vLLM, ..."""

import json
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.pricing.tokens import estimate_prompt_tokens, estimate_tokens
from llm_gateway.providers.base import ErrorKind, ProviderError

# A 400 about context length is not the client's fault: another model in the chain may have more room.
_CONTEXT_HINTS = ("context_length_exceeded", "context length", "maximum context", "too many tokens")


def _classify(status: int, message: str, code: str) -> ErrorKind:
    if status in (400, 413, 422):
        text = f"{code} {message}".lower()
        return "provider" if any(hint in text for hint in _CONTEXT_HINTS) else "client"
    if status in (408, 429) or status >= 500:
        return "transient"
    return "provider"  # 401, 403, 404 (unknown model), ...: our configuration, not the request


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return max(0.0, float(resp.headers["retry-after"]))
    except (KeyError, ValueError):
        return None


class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, api_key: str | None, client: httpx.AsyncClient) -> None:
        self.name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client

    def _body(self, request: ChatCompletionRequest, model: str, stream: bool) -> dict:
        body: dict = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": stream,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            body["response_format"] = request.response_format.model_dump(exclude_none=True)
        if stream:
            body["stream_options"] = {"include_usage": True}  # without it most providers omit `usage`
        return body

    def _http_error(self, resp: httpx.Response) -> ProviderError:
        """Only a `client` error carries the upstream message (it describes the client's own request)."""
        message, code = "", ""
        try:
            err = resp.json().get("error", {})
            message, code = str(err.get("message", "")), str(err.get("code", ""))
        except (ValueError, AttributeError):
            pass
        kind = _classify(resp.status_code, message, code)
        text = f"{self.name}: HTTP {resp.status_code}"
        if kind == "client" and message:
            text = f"{self.name}: {message[:300]}"
        return ProviderError(text, kind=kind, status=resp.status_code, retry_after=_retry_after(resp))

    def _network_error(self, exc: httpx.HTTPError) -> ProviderError:
        return ProviderError(f"{self.name}: {type(exc).__name__}", kind="transient")

    async def complete(self, request: ChatCompletionRequest, model: str) -> dict:
        try:
            resp = await self._client.post(
                self._url, json=self._body(request, model, False), headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if resp.status_code >= 400:
            raise self._http_error(resp)
        try:
            data = resp.json()
            reply = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(f"{self.name}: malformed response", kind="provider") from exc
        if not data.get("usage"):
            prompt, done = estimate_prompt_tokens(request.messages), estimate_tokens(reply)
            data["usage"] = {
                "prompt_tokens": prompt,
                "completion_tokens": done,
                "total_tokens": prompt + done,
            }
        return data

    async def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[dict]:
        seen, last, usage, reply_chars = False, {}, None, 0
        try:
            async with self._client.stream(
                "POST", self._url, json=self._body(request, model, True), headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._http_error(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError as exc:
                        raise ProviderError(f"{self.name}: malformed stream chunk", kind="provider") from exc
                    if "error" in chunk:  # some providers report failures in-band
                        raise ProviderError(f"{self.name}: error in stream", kind="transient")
                    seen, last = True, chunk
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        reply_chars += len((choice.get("delta") or {}).get("content") or "")
                    yield chunk
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if not seen:
            raise ProviderError(f"{self.name}: empty stream", kind="provider")
        if not usage:  # keep the contract: the last chunk has `usage`
            prompt, done = estimate_prompt_tokens(request.messages), estimate_tokens("x" * reply_chars)
            yield {
                "id": last.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion.chunk",
                "created": last.get("created") or int(time.time()),
                "model": last.get("model") or model,
                "choices": [],
                "usage": {"prompt_tokens": prompt, "completion_tokens": done, "total_tokens": prompt + done},
            }
