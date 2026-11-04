"""Shared test setup: an app with known client keys and no real provider reachable."""

import hashlib
import tempfile
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_gateway.main import create_app
from llm_gateway.routing.config import load_routes
from llm_gateway.settings import Settings

FULL_KEY = "gw_full_" + "f" * 32  # every alias
DEMO_KEY = "gw_demo_" + "d" * 32  # only `fast`
NO_ALIASES_KEY = "gw_none_" + "n" * 32

_DIR = Path(tempfile.mkdtemp(prefix="gw-tests-"))


def _sha(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _write_configs() -> tuple[Path, Path]:
    all_aliases = list(load_routes(Path("config/routes.yaml")).aliases)
    clients = {
        "clients": {
            "full": {
                "key_sha256": _sha(FULL_KEY),
                "allowed_aliases": all_aliases,
                "budget_usd_per_day": 5.0,
                "rpm": 100000,
            },
            "demo": {"key_sha256": _sha(DEMO_KEY), "allowed_aliases": ["fast"], "rpm": 100000},
            "none": {"key_sha256": _sha(NO_ALIASES_KEY)},
        }
    }
    providers = yaml.safe_load(Path("config/providers.yaml").read_text(encoding="utf-8"))
    providers["providers"]["ollama"]["api_key_env"] = "GW_TEST_NEVER_SET"  # keeps Ollama unregistered
    clients_file, providers_file = _DIR / "clients.yaml", _DIR / "providers.yaml"
    clients_file.write_text(yaml.safe_dump(clients), encoding="utf-8")
    providers_file.write_text(yaml.safe_dump(providers), encoding="utf-8")
    return clients_file, providers_file


CLIENTS_FILE, PROVIDERS_FILE = _write_configs()


def make_app(env: str = "test", **overrides) -> FastAPI:
    """Only `mock` is registered: no key is set and .env is ignored, so no test touches the network.
    The cache and embedder are off unless a test asks (`cache_backend="memory"`, `embedder="hash"`)."""
    kwargs = {
        "cache_backend": "off",
        "embedder": "off",
        "groq_api_key": None,
        "google_api_key": None,
        "anthropic_api_key": None,
        **overrides,
    }
    settings = Settings(
        _env_file=None, env=env, clients_file=CLIENTS_FILE, providers_file=PROVIDERS_FILE, **kwargs
    )
    return create_app(settings)


def make_client(env: str = "test", key: str | None = FULL_KEY) -> TestClient:
    client = TestClient(make_app(env))
    if key:
        client.headers["Authorization"] = f"Bearer {key}"
    return client
