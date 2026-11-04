"""Structured output: check (and lightly repair) a completion against `response_format` (PLAN §3.2, R15).

`json_object`: the reply must be a JSON object. `json_schema`: it must also satisfy the schema.
Models often wrap JSON in a code fence or add a sentence around it; that is repaired here without a new call.
Anything else is a validation problem the executor answers with one repair retry on the same provider."""

import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from llm_gateway.api.schemas import ChatCompletionRequest, ChatMessage, ResponseFormat

MAX_PROBLEM_CHARS = 300


class InvalidSchema(Exception):
    """The client's own schema is not valid JSON Schema: a 400, found before any provider call."""


def _extract_json(text: str) -> str | None:
    """The JSON text inside `text`: the whole reply, a code fence's body, or the outermost {...} or [...]."""
    text = text.strip()
    candidates = [text]
    if text.startswith("```"):
        body = text.split("\n", 1)[1] if "\n" in text else ""
        candidates.append(body.rsplit("```", 1)[0].strip())
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = text.find(open_), text.rfind(close)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            json.loads(candidate)
        except ValueError:
            continue
        return candidate
    return None


class OutputValidator:
    def __init__(self, fmt: ResponseFormat) -> None:
        self.kind = fmt.type
        self._validator: Draft202012Validator | None = None
        if fmt.type == "json_schema":
            schema = (fmt.json_schema or {}).get("schema")
            if not isinstance(schema, dict):
                raise InvalidSchema("`response_format.json_schema.schema` must be a JSON Schema object.")
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise InvalidSchema(
                    f"`response_format.json_schema.schema` is invalid: {exc.message}"
                ) from exc
            self._validator = Draft202012Validator(schema)

    def check(self, completion: dict) -> str | None:
        """None when the reply is valid (the message content may be rewritten to the clean JSON text).
        Otherwise a short problem statement that is safe to show to the model."""
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if choice.get("finish_reason") == "length":
            return "the reply was cut off before the JSON was complete"
        if not content.strip():
            return "the reply was empty"
        text = _extract_json(content)
        if text is None:
            return "the reply is not valid JSON"
        value: Any = json.loads(text)
        if not isinstance(value, dict | list):
            return "the reply must be a JSON object"
        if self.kind == "json_object" and not isinstance(value, dict):
            return "the reply must be a JSON object"
        if self._validator is not None:
            error = next(iter(sorted(self._validator.iter_errors(value), key=lambda e: list(e.path))), None)
            if error is not None:
                where = "/".join(str(p) for p in error.path) or "(root)"
                return f"at {where}: {error.message}"[:MAX_PROBLEM_CHARS]
        if text != content:
            message["content"] = text
            completion.setdefault("_meta", {})["structured_repaired"] = True
        return None


def build_validator(fmt: ResponseFormat | None) -> OutputValidator | None:
    if fmt is None or fmt.type == "text":
        return None
    return OutputValidator(fmt)


def repair_request(request: ChatCompletionRequest, completion: dict, problem: str) -> ChatCompletionRequest:
    """The same request plus the invalid reply and what was wrong with it (one retry, same provider)."""
    reply = ((completion.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    note = (
        f"Your previous reply was rejected: {problem}. "
        "Answer again with ONLY valid JSON that follows the requested format. No prose and no code fences."
    )
    messages = [
        *request.messages,
        ChatMessage(role="assistant", content=reply or "(empty)"),
        ChatMessage(role="user", content=note),
    ]
    return request.model_copy(update={"messages": messages})


def merge_usage(a: dict | None, b: dict | None) -> dict | None:
    """Usage of two calls as one (the budget settles the sum)."""
    if not a or not b:
        return a or b
    out = {k: int(a.get(k) or 0) + int(b.get(k) or 0) for k in ("prompt_tokens", "completion_tokens")}
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    da, db = a.get("prompt_tokens_details") or {}, b.get("prompt_tokens_details") or {}
    if da or db:
        out["prompt_tokens_details"] = {
            k: int(da.get(k) or 0) + int(db.get(k) or 0) for k in ("cached_tokens", "cache_creation_tokens")
        }
    return out
