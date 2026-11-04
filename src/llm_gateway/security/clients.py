"""Clients and their API keys. Only SHA-256 hashes are stored (config/clients.yaml, git-ignored)."""
import hashlib
import hmac
import logging
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from llm_gateway.routing.config import RoutesConfig

log = logging.getLogger(__name__)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ClientConfig(BaseModel):
    key_sha256: str
    rpm: int = Field(default=60, gt=0)                 # enforced in M8
    budget_usd_per_day: float = Field(default=0.0, ge=0)   # enforced in M5
    allowed_aliases: list[str] = Field(default_factory=list)   # fail-closed: empty means no alias

    @field_validator("key_sha256")
    @classmethod
    def _is_sha256_hex(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("key_sha256 must be 64 hex characters (create keys with scripts/new_key.py)")
        return value


class ClientRegistry:
    def __init__(self, clients: dict[str, ClientConfig]) -> None:
        hashes = [c.key_sha256 for c in clients.values()]
        if len(set(hashes)) != len(hashes):
            raise ValueError("two clients share the same key_sha256")
        self.clients = clients

    def get(self, name: str) -> ClientConfig | None:
        return self.clients.get(name)

    def authenticate(self, key: str) -> str | None:
        """Client name for a raw key, or None. Compares every hash in constant time (no early exit)."""
        digest = hashlib.sha256(key.encode()).hexdigest()
        found = None
        for name, client in self.clients.items():
            if hmac.compare_digest(digest, client.key_sha256):
                found = name
        return found

    def check_routes(self, routes: RoutesConfig) -> None:
        for name, client in self.clients.items():
            unknown = set(client.allowed_aliases) - set(routes.aliases)
            if unknown:
                raise ValueError(f"client {name!r} allows unknown aliases: {sorted(unknown)}")


def load_clients(path: Path) -> ClientRegistry:
    """A missing file gives an empty registry: every request is rejected (fail-closed)."""
    if not path.exists():
        log.warning("%s not found: all requests will be rejected. Create it with scripts/new_key.py", path)
        return ClientRegistry({})
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ClientRegistry({n: ClientConfig.model_validate(c) for n, c in (raw.get("clients") or {}).items()})
