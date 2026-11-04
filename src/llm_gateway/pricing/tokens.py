"""Token estimates. Deliberately conservative (they round up): an underestimate would undercount spend."""

import math
from typing import Any

from llm_gateway.api.schemas import ChatMessage


def content_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if part.get("type") == "text")


def estimate_tokens(text: str) -> int:
    """1 token per 3 characters: more than real tokenizers give for English, so it errs on the safe side."""
    return math.ceil(len(text) / 3)


IMAGE_TOKENS = 800  # flat charge per image: providers bill 250-1600 depending on size; this errs high


def estimate_prompt_tokens(messages: list[ChatMessage]) -> int:
    images = sum(
        1 for m in messages if isinstance(m.content, list) for p in m.content if p.get("type") == "image_url"
    )
    return sum(estimate_tokens(content_text(m.content)) for m in messages) + images * IMAGE_TOKENS
