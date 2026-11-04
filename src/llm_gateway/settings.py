"""Runtime settings (environment variables, prefix GW_)."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GW_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"   # the mock provider only exists in dev and test
    routes_file: Path = Path("config/routes.yaml")
    prices_file: Path = Path("config/prices.yaml")
    cache_file: Path = Path("config/cache.yaml")
    providers_file: Path = Path("config/providers.yaml")
    clients_file: Path = Path("config/clients.yaml")
    max_in_flight: int = 64                       # global load-shedding limit (PLAN §4.7)
    drain_timeout_s: float = 8.0                  # SIGTERM: time active streams get to finish (PLAN §4.9)
    cache_backend: Literal["memory", "off"] = "memory"   # `redis` joins once verified on Redis 8
    embedder: str = "e5-small"                    # `hash` (offline stand-in), a model name, or `off`
    models_dir: Path = Path("models")             # ONNX model files, one folder per embedder (git-ignored)
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    groq_api_key: str | None = None               # free tier ($0)
    google_api_key: str | None = None             # Google AI Studio free tier ($0): Gemma 4
    anthropic_api_key: str | None = None          # only set for P3's final test
    budget_backend: Literal["memory", "redis"] = "memory"   # redis: atomic Lua budget that survives restarts
    global_day_budget_usd: float | None = None    # cap on all clients together per UTC day (paid only)
    run_budget_usd: float = 0.0                   # 0 = paid providers disabled
    log_prompts: bool = False
    guardrails_mode: str = "monitor"              # off | monitor | enforce
