"""OpenAI-compatible request/response schemas (subset implemented by the gateway)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = Field(description="An alias from routes.yaml, e.g. 'fast', 'smart', 'judge'")
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    response_format: ResponseFormat | None = None
    cache: bool | None = None  # gateway extension: opt in/out of caching
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None  # clamped to the alias's `effort`
    # Set by the gateway, never by the client (`exclude`d from dumps; _serve overwrites whatever came in):
    prompt_cache: bool = Field(default=False, exclude=True)


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "llm-gateway"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
