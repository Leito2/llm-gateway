"""Prompt-injection classifier on ONNX Runtime (stretch, PLAN 8.4): Llama Prompt Guard 2 (22M) or any
sequence classifier exported to ONNX with a "malicious" label.

HONEST SCOPE. The code is written and unit-tested against a fake session, but NO model is bundled and this
repository has never run it on real weights: Prompt Guard 2 is a GATED model (accepting Meta's license on
Hugging Face is a personal step) and its ONNX export was not available here. Put `model.onnx` (or
`model_quantized.onnx`), `tokenizer.json` and `config.json` in `models/prompt-guard/` and install
`uv sync --extra semantic`; until then `load_prompt_guard` returns None and only the regex heuristic runs.
Inference runs in a one-thread pool (ADR-6) and a slow or failing classifier counts as "no verdict"."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

MAX_TOKENS = 512
FILES = ("model_quantized.onnx", "model.onnx")
POSITIVE_WORDS = ("malic", "inject", "jailbreak", "unsafe")


def positive_index(config: dict) -> int:
    """The "attack" output: by label name when the config has one, else the second (index 1)."""
    for idx, label in sorted((config.get("id2label") or {}).items(), key=lambda kv: int(kv[0])):
        if any(w in str(label).lower() for w in POSITIVE_WORDS):
            return int(idx)
    return 1


class PromptGuard:
    def __init__(self, session, tokenizer, positive: int, np, timeout_s: float = 1.0) -> None:
        self._session, self._tokenizer, self._positive, self._np = session, tokenizer, positive, np
        self._inputs = {i.name for i in session.get_inputs()}
        self._timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnx-guard")

    def _score_sync(self, text: str) -> float:
        np = self._np
        enc = self._tokenizer.encode(text)
        ids = np.array([enc.ids[:MAX_TOKENS]], dtype=np.int64)
        mask = np.array([enc.attention_mask[:MAX_TOKENS]], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = np.asarray(self._session.run(None, feed)[0], dtype="float64")[0]
        exp = np.exp(logits - logits.max())
        return float((exp / exp.sum())[self._positive])

    async def score(self, text: str) -> float | None:
        """Probability that `text` is an attack, or None when the model gave no verdict (slow, failed)."""
        try:
            async with asyncio.timeout(self._timeout_s):
                return await asyncio.get_running_loop().run_in_executor(self._pool, self._score_sync, text)
        except Exception as exc:  # noqa: BLE001 - TimeoutError included: no verdict, never a failed request
            log.warning("prompt guard gave no verdict (%s)", type(exc).__name__)
            return None


def load_prompt_guard(model_dir: Path) -> PromptGuard | None:
    model = next((model_dir / f for f in FILES if (model_dir / f).exists()), None)
    if model is None or not (model_dir / "tokenizer.json").exists():
        log.info("prompt guard not loaded: model files not found in %s", model_dir)
        return None
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError:
        log.warning("prompt guard unavailable: install the extra with `uv sync --extra semantic`")
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return PromptGuard(session, tokenizer, positive_index(config), np)
