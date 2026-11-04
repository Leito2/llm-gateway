"""Stretch · Langfuse through OTLP/HTTP (its legacy ingestion API is deprecated). Config only: no real Langfuse."""

import base64

from helpers import make_app

from llm_gateway.settings import Settings
from llm_gateway.telemetry.tracing import export_targets


def settings(**kw):
    return Settings(_env_file=None, **kw)


def test_langfuse_keys_become_an_otlp_target_with_basic_auth_and_the_version_header():
    s = settings(
        langfuse_host="https://cloud.langfuse.com/", langfuse_public_key="pk", langfuse_secret_key="sk"
    )
    ((endpoint, headers),) = export_targets(s)
    assert endpoint == "https://cloud.langfuse.com/api/public/otel"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"pk:sk").decode()
    assert headers["x-langfuse-ingestion-version"] == "4"


def test_both_a_collector_and_langfuse_can_be_active_together():
    s = settings(
        otel_endpoint="http://jaeger:4318",
        langfuse_host="https://lf.test",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )
    assert [t[0] for t in export_targets(s)] == ["http://jaeger:4318", "https://lf.test/api/public/otel"]


def test_incomplete_langfuse_settings_export_nothing():
    assert export_targets(settings(langfuse_host="https://lf.test", langfuse_public_key="pk")) == []
    assert export_targets(settings()) == []


def test_the_app_builds_exporters_with_those_endpoints_and_headers(monkeypatch):
    from opentelemetry.sdk.trace.export import SpanExporter

    made = []

    class Fake(SpanExporter):
        def __init__(self, endpoint=None, headers=None, **_):
            made.append((endpoint, headers))

        def export(self, spans):
            return 0

    monkeypatch.setattr("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", Fake)
    app = make_app(langfuse_host="https://lf.test", langfuse_public_key="pk", langfuse_secret_key="sk")
    assert app.state.gateway.tracer.enabled
    ((endpoint, headers),) = made
    assert endpoint == "https://lf.test/api/public/otel/v1/traces"
    assert headers["x-langfuse-ingestion-version"] == "4" and headers["Authorization"].startswith("Basic ")
