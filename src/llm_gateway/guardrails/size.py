"""Size and shape guardrail (PLAN §8.4, core, mode `enforce`): bounds cost and abuse before any call."""

from llm_gateway.api.schemas import ChatMessage
from llm_gateway.routing.config import AliasPolicy


def count_images(messages: list[ChatMessage]) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m.content, list)
        for part in m.content
        if part.get("type") == "image_url"
    )


def check_shape(policy: AliasPolicy, messages: list[ChatMessage]) -> tuple[str, str] | None:
    """(code, message) of the first limit the request breaks, or None."""
    if len(messages) > policy.max_messages:
        return "too_many_messages", f"At most {policy.max_messages} messages per request for this alias."
    images = count_images(messages)
    if images > policy.max_images:
        return "too_many_images", f"At most {policy.max_images} images per request for this alias."
    return None
