"""Runtime settings (environment variables, prefix GW_)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GW_", env_file=".env", extra="ignore")

    routes_file: Path = Path("config/routes.yaml")
    prices_file: Path = Path("config/prices.yaml")
    clients_file: Path = Path("config/clients.yaml")
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    anthropic_api_key: str | None = None          # only set for P3's final test
    run_budget_usd: float = 0.0                   # 0 = paid providers disabled
    log_prompts: bool = False
