"""Create an API key for a client (stdlib only).

Usage: python scripts/new_key.py <client> [alias ...]
       python scripts/new_key.py --admin

Prints the key once and the config/clients.yaml entry that holds its SHA-256 hash. The key itself is
never stored: lose it and you create a new one. Format: gw_<client>_<32 random characters>.
"""
import hashlib
import re
import secrets
import string
import sys

ALPHABET = string.ascii_letters + string.digits


def admin_key() -> int:
    key = "gw_admin_" + "".join(secrets.choice(ALPHABET) for _ in range(40))
    print(f"Admin key (shown once; Prometheus reads it from observability/admin.key): {key}\n")
    print("Put this hash in .env (guards /stats, /metrics and /admin/*):")
    print(f"  GW_ADMIN_KEY_SHA256={hashlib.sha256(key.encode()).hexdigest()}")
    return 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--admin"]:
        return admin_key()
    if len(argv) < 2 or not re.fullmatch(r"[a-z0-9-]+", argv[1]):
        print(__doc__)
        print("Client names use lowercase letters, digits and '-' only (the key prefix must stay parseable).")
        return 2
    client, aliases = argv[1], argv[2:] or ["fast"]
    key = f"gw_{client}_" + "".join(secrets.choice(ALPHABET) for _ in range(32))
    digest = hashlib.sha256(key.encode()).hexdigest()
    print(f"API key (shown once, give it to the client): {key}\n")
    print("Add this under `clients:` in config/clients.yaml (git-ignored):")
    listed = ", ".join(aliases)
    print(f'  {client}: {{ key_sha256: "{digest}", rpm: 60, budget_usd_per_day: 0,')
    print(f"    allowed_aliases: [{listed}] }}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
