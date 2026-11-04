"""Shared pieces of the benchmarks: throwaway configs, subprocess servers, an open-loop load generator."""

import asyncio
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
KEY = "gw_bench_" + "b" * 32
RESULTS = ROOT / "bench" / "results"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_configs(upstream_port: int, directory: Path) -> dict[str, str]:
    """routes/providers/clients for the benchmark. Aliases: bench (no cache), bench_exact, bench_sem."""
    chain = [{"provider": "upstream", "model": "bench-model"}]
    base = {"chain": chain, "max_tokens": 64, "ttft_timeout_s": 5, "total_timeout_s": 30}
    routes = {
        "aliases": {
            "bench": {**base, "cache": False, "semantic_cache": False},
            "bench_exact": {**base, "cache": True, "semantic_cache": False, "cache_ttl_s": 3600},
            "bench_sem": {**base, "cache": True, "semantic_cache": True, "cache_ttl_s": 3600},
        }
    }
    providers = {
        "providers": {
            "upstream": {"kind": "openai_compat", "base_url": f"http://127.0.0.1:{upstream_port}/v1"}
        }
    }
    clients = {
        "clients": {
            "bench": {
                "key_sha256": hashlib.sha256(KEY.encode()).hexdigest(),
                "rpm": 10_000_000,
                "allowed_aliases": list(routes["aliases"]),
            }
        }
    }
    paths = {}
    for name, data in (("routes", routes), ("providers", providers), ("clients", clients)):
        path = directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        paths[name] = str(path)
    return paths


def wait_port(port: int, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} never opened")


@contextmanager
def server(app: str, port: int, env: dict[str, str] | None = None, workers: int = 1):
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app,
        "--port",
        str(port),
        "--log-level",
        "warning",
        "--workers",
        str(workers),
        "--no-access-log",
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), **(env or {})})
    try:
        wait_port(port)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


@contextmanager
def gateway_stack(embedder: str = "off", workers: int = 1):
    """Upstream + gateway as separate processes. Yields (gateway_url, upstream_url)."""
    up_port, gw_port = free_port(), free_port()
    with tempfile.TemporaryDirectory(prefix="gw-bench-") as tmp:
        paths = write_configs(up_port, Path(tmp))
        env = {
            "GW_ENV": "prod",
            "GW_ROUTES_FILE": paths["routes"],
            "GW_PROVIDERS_FILE": paths["providers"],
            "GW_CLIENTS_FILE": paths["clients"],
            "GW_ACCESS_LOG": "false",
            "GW_EMBEDDER": embedder,
            "GW_CACHE_BACKEND": "memory",
            "GW_MAX_IN_FLIGHT": "100000",
        }
        with server("bench.upstream:app", up_port), server("llm_gateway.main:app", gw_port, env, workers):
            yield f"http://127.0.0.1:{gw_port}", f"http://127.0.0.1:{up_port}"


def body(alias: str, text: str) -> dict:
    return {"model": alias, "messages": [{"role": "user", "content": text}], "max_tokens": 16}


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    if not s:
        return {}

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "n": len(s),
        "p50": round(q(0.50), 2),
        "p95": round(q(0.95), 2),
        "p99": round(q(0.99), 2),
        "mean": round(statistics.fmean(s), 2),
    }


async def open_loop(
    url: str, make_request: Callable[[int], dict], rate: float, seconds: float, headers: dict | None = None
) -> dict:
    """Fire `rate` requests per second on a fixed schedule, whatever the server does (latency counts from the
    scheduled send time, so a slow server cannot hide its own queueing). Returns percentiles in ms."""
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    latencies: list[float] = []
    errors = 0
    async with httpx.AsyncClient(timeout=30, limits=limits, headers=headers) as client:

        async def one(i: int, due: float) -> None:
            nonlocal errors
            try:
                r = await client.post(url, json=make_request(i))
                if r.status_code != 200:
                    errors += 1
                    return
            except httpx.HTTPError:
                errors += 1
                return
            latencies.append((time.perf_counter() - due) * 1000)

        start = time.perf_counter()
        tasks = []
        for i in range(int(rate * seconds)):
            due = start + i / rate
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i, due)))
        await asyncio.gather(*tasks)
    return {**percentiles(latencies), "errors": errors, "rate": rate}


def save(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
