"""Runtime settings (environment variables, prefix GW_)."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GW_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"  # the mock provider only exists in dev and test
    routes_file: Path = Path("config/routes.yaml")
    prices_file: Path = Path("config/prices.yaml")
    cache_file: Path = Path("config/cache.yaml")
    providers_file: Path = Path("config/providers.yaml")
    clients_file: Path = Path("config/clients.yaml")
    max_in_flight: int = 64  # global load-shedding limit (PLAN §4.7)
    stream_ping_s: float = 10.0  # SSE keep-alive comment after this much silence (R10)
    drain_timeout_s: float = 8.0  # SIGTERM: time active streams get to finish (PLAN §4.9)
    cache_backend: Literal["memory", "redis", "off"] = "memory"  # redis: shared, survives restarts
    embedder: str = "e5-small"  # `hash` (offline stand-in), a model name, or `off`
    models_dir: Path = Path("models")  # ONNX model files, one folder per embedder (git-ignored)
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    groq_api_key: str | None = None  # free tier ($0)
    google_api_key: str | None = None  # Google AI Studio free tier ($0): Gemma 4
    anthropic_api_key: str | None = None  # only set for P3's final test
    budget_backend: Literal["memory", "redis"] = "memory"  # redis: atomic Lua budget that survives restarts
    global_day_budget_usd: float | None = None  # cap on all clients together per UTC day (paid only)
    run_budget_usd: float = 0.0  # 0 = paid providers disabled
    log_prompts: bool = False  # R16/R22: needs Presidio, otherwise the gateway will not start
    access_log: bool = True  # one JSON line per request on stdout (never prompts)
    admin_key_sha256: str | None = None  # SHA-256 of the admin key: guards /stats, /metrics, /admin/*
    otel_endpoint: str | None = None  # OTLP/HTTP endpoint (e.g. http://jaeger:4318); unset = no tracing
    rate_limit_backend: Literal["memory", "redis"] = "memory"  # redis falls back to memory when it is down
    cors_origins: str = ""  # comma-separated origins (or *) allowed to call from a browser
    max_body_bytes: int = 8_000_000  # larger requests get 413 before parsing
    guardrails_mode: Literal["off", "monitor", "enforce"] = "monitor"  # injection/PII heuristics
    reload_interval_s: float = 0.0  # >0: reload routes/clients/prices when their files change
