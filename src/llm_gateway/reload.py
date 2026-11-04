"""Hot reload of routes, client keys and prices (stretch, PLAN 3 and 8.1).

Everything is loaded and validated first; only then are the three references swapped, so a bad edit leaves
the old configuration serving (all or nothing). Requests already running keep the policy object they began
with. Not reloaded: providers and their keys, quotas, cache, budget caps (those need a restart)."""

import asyncio
import logging
from pathlib import Path

from llm_gateway.pricing.prices import load_prices
from llm_gateway.providers.config import load_providers
from llm_gateway.routing.config import load_routes
from llm_gateway.security.clients import load_clients
from llm_gateway.state import GatewayState

log = logging.getLogger("gateway.reload")


def reload_config(gw: GatewayState) -> dict:
    """Raises on any invalid file; the running configuration is untouched in that case."""
    s = gw.settings
    routes = load_routes(s.routes_file)
    load_providers(s.providers_file).check_routes(routes)
    clients = load_clients(s.clients_file)
    clients.check_routes(routes)
    prices = load_prices(s.prices_file)
    gw.routes, gw.clients, gw.prices = routes, clients, prices
    if gw.budget is not None:
        gw.budget.prices = prices
    summary = {"aliases": sorted(routes.aliases), "clients": len(clients.clients)}
    log.info("configuration reloaded: %s", summary)
    return summary


def _stamp(gw: GatewayState) -> tuple:
    s = gw.settings
    paths: list[Path] = [s.routes_file, s.clients_file, s.prices_file, s.providers_file]
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)


async def watch(gw: GatewayState, interval_s: float) -> None:
    """Reload when a config file's mtime changes. A failed reload is logged and retried on the next edit."""
    seen = _stamp(gw)
    while True:
        await asyncio.sleep(interval_s)
        now = _stamp(gw)
        if now == seen:
            continue
        seen = now
        try:
            reload_config(gw)
        except Exception as exc:  # noqa: BLE001 - any bad file must keep the old config, not kill the task
            log.error("configuration reload rejected, keeping the old one: %s", exc)
