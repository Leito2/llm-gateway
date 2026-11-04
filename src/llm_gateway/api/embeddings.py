"""POST /v1/embeddings: the gateway's own embedder, OpenAI-compatible (PLAN M4)."""

import asyncio
import base64
import struct
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from llm_gateway.api.errors import GatewayError
from llm_gateway.pricing.tokens import estimate_tokens
from llm_gateway.security.auth import require_client

router = APIRouter()

MAX_INPUTS = 64
MAX_CHARS = 8000
TIMEOUT_S = 10.0


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"


@router.post("/v1/embeddings")
async def embeddings(
    body: EmbeddingRequest, request: Request, client: Annotated[str, Depends(require_client)]
) -> dict:
    gw = request.app.state.gateway
    embedder = gw.embedder
    if embedder is None:
        gw.stats.record("embeddings", "-", 503)
        raise GatewayError(
            503,
            "embeddings_unavailable",
            "No embedder is configured (GW_EMBEDDER is `off` or its model is missing).",
            type_="server_error",
        )
    if body.model != embedder.name:
        gw.stats.record("embeddings", "-", 404)
        raise GatewayError(
            404,
            "model_not_found",
            f"The embedding model `{body.model}` does not exist. Available: {embedder.name}.",
            param="model",
        )
    texts = [body.input] if isinstance(body.input, str) else body.input
    if not texts or len(texts) > MAX_INPUTS or any(not t.strip() or len(t) > MAX_CHARS for t in texts):
        raise GatewayError(
            400,
            "invalid_request",
            f"`input` needs 1 to {MAX_INPUTS} non-empty strings of at most {MAX_CHARS} characters.",
            param="input",
        )
    try:
        async with asyncio.timeout(TIMEOUT_S):
            vectors = await embedder.embed(texts)
    except TimeoutError as exc:
        gw.stats.record("embeddings", "-", 504)
        raise GatewayError(
            504, "upstream_timeout", "The embedder did not answer in time.", type_="server_error"
        ) from exc
    gw.stats.record("embeddings", embedder.name, 200)
    tokens = sum(estimate_tokens(t) for t in texts)
    return {
        "object": "list",
        "model": embedder.name,
        "data": [
            {"object": "embedding", "index": i, "embedding": _encode(v, body.encoding_format)}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


def _encode(vector: list[float], fmt: str) -> list[float] | str:
    if fmt == "float":
        return vector
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
