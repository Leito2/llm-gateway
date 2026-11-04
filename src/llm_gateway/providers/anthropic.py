"""Native adapter for Claude Haiku 5.5 over the Messages API (ADR-10, ADR-0005).

Why native and not the OpenAI-compatible adapter: prompt caching (`cache_control`), thinking (billed as
output, never forwarded) and refusals (`stop_reason: "refusal"`) are under our control.

Translation rules, all from ADR-0005:
- `temperature`, `top_p` and `top_k` are dropped (the API rejects non-default values);
  `X-Params-Dropped` lists them.
- `reasoning_effort` (already clamped to the alias by the gateway) becomes `output_config.effort`.
- The system prompt gets `cache_control` when the alias enables prompt caching and it reaches the model's
  512-token minimum.
- `usage` follows the OpenAI convention: `prompt_tokens` INCLUDES cache reads and writes, which are reported
  in `prompt_tokens_details` so the budget can price them apart.
- `refusal` -> `finish_reason: "content_filter"`. It is a normal answer: no fallback, not cached."""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.api.schemas import ChatCompletionRequest, ChatMessage
from llm_gateway.pricing.tokens import estimate_tokens
from llm_gateway.providers.base import ErrorKind, ProviderError

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
CACHE_MIN_TOKENS = 512  # smallest prefix the model will cache
STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_CONTEXT_HINTS = ("prompt is too long", "too many tokens", "context window", "exceed")


def classify(status: int, message: str, error_type: str) -> ErrorKind:
    if status == 400 or error_type == "invalid_request_error":
        text = message.lower()
        return "provider" if any(h in text for h in _CONTEXT_HINTS) else "client"
    if status == 413:
        return "client"
    if status in (408, 429, 529) or status >= 500 or error_type in ("overloaded_error", "rate_limit_error"):
        return "transient"
    return "provider"  # 401, 403, 404: our configuration, not the request


def _image_block(part: dict[str, Any]) -> dict[str, Any] | None:
    url = (part.get("image_url") or {}).get("url", "") if isinstance(part.get("image_url"), dict) else ""
    if url.startswith("data:") and ";base64," in url:
        header, data = url.split(",", 1)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": header[5:].split(";")[0], "data": data},
        }
    if url:
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    out: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "text" and part.get("text"):
            out.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            block = _image_block(part)
            if block:
                out.append(block)
    return out


def split_messages(
    messages: list[ChatMessage], prompt_cache: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(system blocks, conversation). Consecutive turns of one role are merged: the API wants alternation."""
    system: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system.extend(_blocks(m.content))
            continue
        role = "assistant" if m.role == "assistant" else "user"  # `tool` results are sent as user turns
        blocks = _blocks(m.content)
        if not blocks:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})
    if prompt_cache and system:
        size = sum(estimate_tokens(b.get("text", "")) for b in system)
        if size >= CACHE_MIN_TOKENS:
            system[-1] = {**system[-1], "cache_control": {"type": "ephemeral"}}
    if not turns:  # a system-only request still needs a user turn
        turns = [{"role": "user", "content": [{"type": "text", "text": "."}]}]
    return system, turns


def normalize_usage(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Anthropic's `input_tokens` excludes cache reads and writes; the OpenAI convention includes them."""
    raw = raw or {}
    plain = int(raw.get("input_tokens") or 0)
    read = int(raw.get("cache_read_input_tokens") or 0)
    write = int(raw.get("cache_creation_input_tokens") or 0)
    prompt, done = plain + read + write, int(raw.get("output_tokens") or 0)
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": done,
        "total_tokens": prompt + done,
    }
    if read or write:
        usage["prompt_tokens_details"] = {"cached_tokens": read, "cache_creation_tokens": write}
    return usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, client: httpx.AsyncClient, base_url: str = DEFAULT_BASE_URL) -> None:
        self._url = base_url.rstrip("/") + "/v1/messages"
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        self._client = client

    # --- request ---------------------------------------------------------------------------------

    def _body(self, request: ChatCompletionRequest, model: str, stream: bool) -> tuple[dict, dict]:
        system, turns = split_messages(request.messages, request.prompt_cache)
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens or 1024,
            "messages": turns,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if request.response_format is not None and request.response_format.type != "text":
            body["system"] = [*system, {"type": "text", "text": _json_instruction(request)}]
        if request.reasoning_effort:
            body["output_config"] = {"effort": request.reasoning_effort}
        dropped = [
            name
            for name in ("temperature", "top_p")
            if name in request.model_fields_set and getattr(request, name) is not None
        ]
        meta: dict[str, Any] = {}
        if dropped:
            meta["params_dropped"] = dropped
        if request.reasoning_effort:
            meta["effort"] = request.reasoning_effort
        return body, meta

    # --- errors ----------------------------------------------------------------------------------

    def _http_error(self, resp: httpx.Response) -> ProviderError:
        message, etype = "", ""
        try:
            err = resp.json().get("error", {})
            message, etype = str(err.get("message", "")), str(err.get("type", ""))
        except (ValueError, AttributeError):
            pass
        kind = classify(resp.status_code, message, etype)
        text = f"{self.name}: HTTP {resp.status_code}"
        if kind == "client" and message:
            text = f"{self.name}: {message[:300]}"
        retry_after = None
        try:
            retry_after = max(0.0, float(resp.headers["retry-after"]))
        except (KeyError, ValueError):
            pass
        return ProviderError(text, kind=kind, status=resp.status_code, retry_after=retry_after)

    def _network_error(self, exc: httpx.HTTPError) -> ProviderError:
        return ProviderError(f"{self.name}: {type(exc).__name__}", kind="transient")

    # --- non-streaming ---------------------------------------------------------------------------

    async def complete(self, request: ChatCompletionRequest, model: str) -> dict:
        body, meta = self._body(request, model, False)
        try:
            resp = await self._client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if resp.status_code >= 400:
            raise self._http_error(resp)
        try:
            data = resp.json()
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            finish = STOP_REASONS.get(data.get("stop_reason") or "end_turn", "stop")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"{self.name}: malformed response", kind="provider") from exc
        completion = {
            "id": f"chatcmpl-{(data.get('id') or uuid.uuid4().hex)[-24:]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model") or model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
            ],
            "usage": normalize_usage(data.get("usage")),
        }
        if meta:
            completion["_meta"] = meta
        return completion

    # --- streaming -------------------------------------------------------------------------------

    async def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[dict]:
        body, meta = self._body(request, model, True)
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }
        usage: dict[str, Any] = {}
        finish = "stop"
        first_sent = False

        def chunk(delta: dict | None, finish_reason: str | None = None, **extra: Any) -> dict:
            choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
            return {**base, "choices": choices, **extra}

        try:
            async with self._client.stream("POST", self._url, json=body, headers=self._headers) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._http_error(resp)
                event = ""
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                    except ValueError as exc:
                        raise ProviderError(f"{self.name}: malformed stream event", kind="provider") from exc
                    kind = data.get("type") or event
                    if kind == "message_start":
                        message = data.get("message") or {}
                        base["id"] = f"chatcmpl-{(message.get('id') or uuid.uuid4().hex)[-24:]}"
                        base["model"] = message.get("model") or model
                        usage.update(message.get("usage") or {})
                        first = chunk({"role": "assistant", "content": ""})
                        if meta:
                            first["_meta"] = meta
                        first_sent = True
                        yield first
                    elif kind == "content_block_delta":
                        delta = data.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield chunk({"content": delta["text"]})
                        else:  # thinking / signature / tool-input deltas: proof of life, nothing to forward
                            yield chunk(None)
                    elif kind == "ping":
                        yield chunk(None)
                    elif kind == "message_delta":
                        delta = data.get("delta") or {}
                        finish = STOP_REASONS.get(delta.get("stop_reason") or "end_turn", "stop")
                        usage.update(data.get("usage") or {})
                    elif kind == "message_stop":
                        yield chunk({}, finish, usage=normalize_usage(usage))
                        return
                    elif kind == "error":
                        err = data.get("error") or {}
                        etype = str(err.get("type", ""))
                        raise ProviderError(
                            f"{self.name}: {etype or 'error'} in stream",
                            kind=classify(500 if etype == "api_error" else 0, "", etype),
                        )
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        raise ProviderError(
            f"{self.name}: stream ended without message_stop" if first_sent else f"{self.name}: empty stream",
            kind="transient" if first_sent else "provider",
        )


def _json_instruction(request: ChatCompletionRequest) -> str:
    """The Messages API has no `response_format`: say it in the system prompt (it still validates)."""
    fmt = request.response_format
    if fmt is not None and fmt.type == "json_schema":
        schema = json.dumps((fmt.json_schema or {}).get("schema", {}), ensure_ascii=False)
        return f"Reply with ONLY a JSON value that satisfies this JSON Schema, with no prose:\n{schema}"
    return "Reply with ONLY a valid JSON object, with no prose."
