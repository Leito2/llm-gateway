"""multilingual-e5-small on ONNX Runtime (int8, CPU): the default embedder (PLAN §6.6).

Needs the `semantic` extra (`uv sync --extra semantic`) and the files in `models/e5-small/`
(`model_quantized.onnx`, `tokenizer.json`). Inference runs in a small bounded thread pool so it never blocks
the event loop and never starves it of CPU (ADR-6)."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

MAX_TOKENS = 128  # cache questions are short; longer text is truncated for matching only
MODEL_FILE = "model_quantized.onnx"


class OnnxEmbedder:
    dim = 384

    def __init__(self, model_dir: Path, name: str = "e5-small", threads: int = 2) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        self._np = np
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model_dir / MODEL_FILE), opts, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")  # XLM-R pad id
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnx-embed")  # bounded: ADR-6

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        np = self._np
        # e5 was trained with a "query: " prefix on both sides of a symmetric pair like ours.
        encoded = self._tokenizer.encode_batch([f"query: {t}" for t in texts])
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feed)[0]  # (batch, tokens, 384)
        weights = mask[:, :, None].astype(hidden.dtype)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled.astype("float32").tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._embed_sync, texts)


def load_onnx_embedder(name: str, model_dir: Path) -> OnnxEmbedder | None:
    """None (with a log line) when the extra or the model files are missing: the semantic layer stays off."""
    if not (model_dir / MODEL_FILE).exists() or not (model_dir / "tokenizer.json").exists():
        log.warning("embedder %r unavailable: model files not found in %s", name, model_dir)
        return None
    try:
        return OnnxEmbedder(model_dir, name)
    except ImportError:
        log.warning("embedder %r unavailable: install the extra with `uv sync --extra semantic`", name)
        return None
