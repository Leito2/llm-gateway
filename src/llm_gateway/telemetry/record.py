"""One record per chat request: feeds the JSON access log, the Prometheus metrics and the request span.

Fields are counts, names and timings. Prompt and answer text never enter a record (R16)."""

import json
import logging
import sys
import time
from contextvars import ContextVar

from llm_gateway.pricing.prices import Usage, cost_micro

access_log = logging.getLogger("gateway.access")
current_record: ContextVar["RequestRecord | None"] = ContextVar("current_record", default=None)


def configure_access_log(enabled: bool) -> None:
    """Stdout, message only, so Cloud Run and `docker logs` get raw JSON lines. Idempotent."""
    access_log.disabled = not enabled
    if enabled and not any(getattr(h, "_gateway", False) for h in access_log.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._gateway = True  # type: ignore[attr-defined]
        access_log.addHandler(handler)
        access_log.setLevel(logging.INFO)


class RequestRecord:
    def __init__(self, gw, request_id: str, client: str, alias: str, stream: bool, span) -> None:
        self.gw, self.request_id, self.client, self.alias, self.stream = gw, request_id, client, alias, stream
        self.span = span
        self.start = time.monotonic()
        self.ttft_s: float | None = None
        self.provider, self.model = "-", ""
        self.depth = 0
        self.cache_layer: str | None = None
        self.status = 200
        self.error_code: str | None = None
        self.usage: dict | None = None
        self.stream_outcome: str | None = None
        self.prompt_text: str | None = None  # only set when LOG_PROMPTS=true and Presidio redacted it
        self._done = False

    def answered(self, provider: str, model: str, depth: int) -> None:
        self.provider, self.model, self.depth = provider, model, depth
        self.ttft_s = time.monotonic() - self.start
        if self.gw.metrics and provider != "cache":
            self.gw.metrics.ttft.labels(self.alias, provider).observe(self.ttft_s)

    def finish(self, status: int | None = None, error_code: str | None = None) -> None:
        if self._done:
            return
        self._done = True
        if status is not None:
            self.status = status
        self.error_code = error_code or self.error_code
        duration = time.monotonic() - self.start
        cost = 0
        parsed = Usage.from_completion(self.usage)
        if parsed is not None and self.gw.prices is not None:
            cost = cost_micro(self.gw.prices.price(self.provider, self.model), parsed)
        m = self.gw.metrics
        if m:
            m.duration.labels(self.alias).observe(duration)
            m.usage(self.provider, self.usage)
            if self.stream and self.stream_outcome:
                m.stream_outcomes.labels(self.stream_outcome).inc()
        line = {
            "ts": round(time.time(), 3),
            "request_id": self.request_id,
            "client": self.client,
            "alias": self.alias,
            "provider": self.provider,
            "model": self.model,
            "attempts": self.depth,
            "cache": self.cache_layer,
            "status": self.status,
            "error": self.error_code,
            "stream": self.stream,
            "stream_outcome": self.stream_outcome,
            "ttft_ms": None if self.ttft_s is None else round(self.ttft_s * 1000, 1),
            "duration_ms": round(duration * 1000, 1),
            "tokens_in": (self.usage or {}).get("prompt_tokens"),
            "tokens_out": (self.usage or {}).get("completion_tokens"),
            "cost_usd": cost / 1_000_000,
            "trace_id": self.span.trace_id,
        }
        if self.prompt_text is not None:
            line["prompt_redacted"] = self.prompt_text
        access_log.info(json.dumps(line, separators=(",", ":")))
        self.span.set(
            {
                "gen_ai.system": self.provider,
                "gen_ai.response.model": self.model or None,
                "gen_ai.usage.input_tokens": (self.usage or {}).get("prompt_tokens"),
                "gen_ai.usage.output_tokens": (self.usage or {}).get("completion_tokens"),
                "gateway.cache": self.cache_layer,
                "http.status_code": self.status,
            }
        )
        self.span.end()
