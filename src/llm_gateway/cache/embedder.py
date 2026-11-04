"""Text → vector. The semantic cache and `/v1/embeddings` depend on this interface only.

`HashEmbedder` is a deterministic stand-in (character n-grams hashed into a fixed space): it needs no
model, so tests, dev and CI run offline. It captures spelling overlap, not meaning, and is never
calibrated for serving.
Real embedders (multilingual-e5-small in ONNX, PLAN §6.6) plug in behind the same Protocol."""

import hashlib
import math
import re
from typing import Protocol

_WORD = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    name: str  # goes into the cache namespace: vectors of different models never mix
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One L2-normalised vector per text, so cosine similarity is a dot product."""
        ...


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = _WORD.findall(text.lower())
        grams = [w for w in words] + [
            f"{w[i : i + 3]}" for w in (f" {w} " for w in words) for i in range(max(1, len(w) - 2))
        ]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def build_embedder(name: str) -> Embedder | None:
    """None means "no embedder": semantic layer and `/v1/embeddings` are off; the exact cache still works."""
    if name == "hash":
        return HashEmbedder()
    return None
