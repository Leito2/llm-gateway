"""Real multilingual-e5-small (ONNX int8). Skipped when the model files or the `semantic` extra are missing,
so CI without the 120 MB download stays green; `uv sync --extra semantic` plus the model enables them."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import make_app, make_client

from llm_gateway.cache.embedder import build_embedder, cosine

MODELS = Path("models")
pytestmark = pytest.mark.skipif(
    not (MODELS / "e5-small" / "model_quantized.onnx").exists() or build_embedder("e5-small", MODELS) is None,
    reason="e5-small model files or the `semantic` extra are not installed",
)


async def test_vectors_are_384_dim_and_normalised():
    e = build_embedder("e5-small", MODELS)
    vecs = await e.embed(["hola", "hello world, how are you today?"])
    assert len(vecs) == 2 and all(len(v) == 384 for v in vecs)
    assert all(abs(cosine(v, v) - 1) < 1e-4 for v in vecs)


async def test_paraphrases_across_languages_rank_above_unrelated_questions():
    e = build_embedder("e5-small", MODELS)
    a, same, other = await e.embed(
        ["How do I reset my password?", "¿Cómo restablezco mi contraseña?", "How long does shipping take?"]
    )
    assert cosine(a, same) > cosine(a, other) + 0.03


async def test_batch_and_single_calls_agree():
    e = build_embedder("e5-small", MODELS)
    (single,) = await e.embed(["short"])
    batch = await e.embed(["short", "a much longer sentence that forces padding in the batch"])
    # Not exactly 1: dynamic int8 quantization scales activations per batch (the cache embeds one text).
    assert cosine(single, batch[0]) > 0.99


def test_the_embeddings_endpoint_serves_the_real_model():
    client = TestClient(make_app(embedder="e5-small", models_dir=MODELS), headers=make_client().headers)
    r = client.post("/v1/embeddings", json={"model": "e5-small", "input": "hola"})
    assert r.status_code == 200 and len(r.json()["data"][0]["embedding"]) == 384
