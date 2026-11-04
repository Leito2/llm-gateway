"""Minimal OpenTelemetry tracing (PLAN §8.2): off unless `GW_OTEL_ENDPOINT` is set (or a provider is given).

One span per request (`gateway.request`, ends when the response ends, streams included) and
one child span per provider attempt, so fallbacks show up as siblings. The W3C `traceparent` header continues
the caller's trace. Attributes follow the GenAI conventions; prompt text is never recorded (R16)."""

import contextlib
import logging
from collections.abc import Iterator, Mapping

from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

log = logging.getLogger(__name__)
_propagator = TraceContextTextMapPropagator()


class ReqSpan:
    """The request span plus the context token to detach. `end` is idempotent."""

    def __init__(self, span: trace.Span | None, token: object | None) -> None:
        self._span, self._token = span, token

    @property
    def trace_id(self) -> str | None:
        if self._span is None:
            return None
        return format(self._span.get_span_context().trace_id, "032x")

    def set(self, attrs: Mapping[str, object]) -> None:
        if self._span is not None:
            for k, v in attrs.items():
                if v is not None:
                    self._span.set_attribute(k, v)  # type: ignore[arg-type]

    def end(self, error: BaseException | None = None) -> None:
        span, self._span = self._span, None
        if span is None:
            return
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
        span.end()

    def detach(self) -> None:
        """Take the span off the current context (call in the task that attached it)."""
        token, self._token = self._token, None
        if token is not None:
            with contextlib.suppress(Exception):
                context.detach(token)


class Tracer:
    def __init__(self, provider: TracerProvider | None = None) -> None:
        self._tracer = provider.get_tracer("llm-gateway") if provider is not None else None

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    def start_request(self, alias: str, headers: Mapping[str, str], request_id: str) -> ReqSpan:
        if self._tracer is None:
            return ReqSpan(None, None)
        parent = _propagator.extract(dict(headers))
        span = self._tracer.start_span(
            "gateway.request",
            context=parent,
            attributes={"gen_ai.request.model": alias, "gateway.request_id": request_id},
        )
        token = context.attach(trace.set_span_in_context(span, parent))
        return ReqSpan(span, token)

    @contextlib.contextmanager
    def attempt(self, provider: str, model: str) -> Iterator[trace.Span | None]:
        if self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(
            "provider.attempt",
            attributes={"gen_ai.system": provider, "gen_ai.request.model": model},
        ) as span:
            try:
                yield span
            except BaseException as exc:
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise


def build_tracer(endpoint: str | None, provider: TracerProvider | None = None) -> Tracer:
    if provider is None and endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.warning("GW_OTEL_ENDPOINT is set but the OTLP exporter is missing: uv sync --extra otel")
            return Tracer()
        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
        )
    return Tracer(provider)
