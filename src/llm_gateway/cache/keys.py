"""What identifies a cache entry: namespace, exact key, and the text that gets embedded (PLAN §6.2)."""

import hashlib
import json
import re
import unicodedata
from typing import Literal

from llm_gateway.api.schemas import ChatMessage
from llm_gateway.pricing.tokens import content_text

SemanticScope = Literal["single", "last_turn"]

_SPACES = re.compile(r"\s+")
_TAG = re.compile(r"[A-Za-z0-9:_.\-/]{1,128}")
MAX_TAGS = 20


def normalize(text: str) -> str:
    """Unicode-compose and collapse whitespace, so `"hola  mundo\\n"` and `"hola mundo"` are one question.
    Case is kept: `US` and `us` are different words."""
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _sha(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def has_non_text(messages: list[ChatMessage]) -> bool:
    """Images and other parts are not in the key, so a request that carries them is never cached."""
    return any(
        not isinstance(m.content, str) and any(p.get("type") != "text" for p in m.content) for m in messages
    )


def system_hash(messages: list[ChatMessage]) -> str:
    return _sha([normalize(content_text(m.content)) for m in messages if m.role == "system"])[:16]


def namespace_base(client: str, alias: str, system: str, prompt_version: str, response_format: object) -> str:
    """Everything except the embedder. Different system prompts, clients or output formats never share."""
    return _sha(client, alias, system, prompt_version, response_format)[:32]


def semantic_namespace(base: str, embedder: str) -> str:
    return _sha(base, embedder)[:32]


def exact_key(base: str, messages: list[ChatMessage]) -> str:
    """The whole conversation, roles included, on top of the namespace base."""
    return _sha(base, [[m.role, normalize(content_text(m.content))] for m in messages])


def semantic_text(messages: list[ChatMessage], scope: SemanticScope) -> str | None:
    """The question to embed, or None when semantic matching would be unsafe for this conversation.

    `single`: exactly one user message and no assistant turns (nothing earlier can change its meaning).
    `last_turn`: every user question stands alone (FAQ style), so the last one is enough."""
    users = [m for m in messages if m.role == "user"]
    if not users:
        return None
    if scope == "single" and (len(users) != 1 or any(m.role in ("assistant", "tool") for m in messages)):
        return None
    text = normalize(content_text(users[-1].content))
    return text or None


def parse_tags(header: str | None) -> tuple[str, ...]:
    """`X-Cache-Tags: doc:1, doc:7`. Unsafe or oversized tags are dropped, not rejected: tags are a hint."""
    if not header:
        return ()
    tags = [t.strip() for t in header.split(",")]
    return tuple(dict.fromkeys(t for t in tags if _TAG.fullmatch(t)))[:MAX_TAGS]
