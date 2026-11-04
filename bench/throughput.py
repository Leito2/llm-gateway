"""G13: throughput of the exact-cache path with 1, 2 and 4 uvicorn workers.

    uv run python -m bench.throughput [--workers 1 2 4] [--seconds 10] [--clients 4]

Closed loop: several client PROCESSES (so the Python load generator is not the only bottleneck) hammer one
cached question. Reports sustained requests/second on THIS machine. The Go predecessor's figure and the
PLAN's '10K+ req/s' claim are not reproduced by a Python gateway and a Python load generator on Windows;
the number is published as it comes out. Note ADR-0003: production runs 1 worker; more workers only matter
for this measurement because each worker has its own in-memory cache."""

import argparse
import asyncio
import multiprocessing as mp
import time

import httpx

from bench.common import KEY, body, gateway_stack, save


async def hammer(url: str, seconds: float, connections: int) -> tuple[int, int]:
    done = errors = 0
    deadline = time.perf_counter() + seconds
    payload = body("bench_exact", "what is the capital of france")
    limits = httpx.Limits(max_connections=connections, max_keepalive_connections=connections)
    async with httpx.AsyncClient(timeout=30, limits=limits, headers={"Authorization": f"Bearer {KEY}"}) as c:

        async def loop() -> None:
            nonlocal done, errors
            while time.perf_counter() < deadline:
                try:
                    r = await c.post(url, json=payload)
                    done += r.status_code == 200
                    errors += r.status_code != 200
                except httpx.HTTPError:
                    errors += 1

        await asyncio.gather(*(loop() for _ in range(connections)))
    return done, errors


def client_process(url: str, seconds: float, connections: int, out) -> None:
    out.put(asyncio.run(hammer(url, seconds, connections)))


def measure(url: str, seconds: float, clients: int, connections: int) -> dict:
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=client_process, args=(url, seconds, connections, q)) for _ in range(clients)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()
    done = sum(r[0] for r in results)
    return {"requests": done, "errors": sum(r[1] for r in results), "rps": round(done / seconds)}


def main(worker_counts: list[int], seconds: float, clients: int) -> dict:
    out: dict = {"seconds": seconds, "client_processes": clients, "workers": {}}
    for workers in worker_counts:
        with gateway_stack(embedder="off", workers=workers) as (gw, _up):
            url = f"{gw}/v1/chat/completions"
            warm = httpx.Client(headers={"Authorization": f"Bearer {KEY}"}, timeout=30)
            for _ in range(200):  # fill the cache of every worker
                warm.post(url, json=body("bench_exact", "what is the capital of france"))
            out["workers"][str(workers)] = measure(url, seconds, clients, connections=16)
    return out


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--clients", type=int, default=4)
    args = ap.parse_args()
    print(save("g13_throughput", main(args.workers, args.seconds, args.clients)).read_text(encoding="utf-8"))
