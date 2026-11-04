"""Langfuse export (stretch, opt-in): one trace + one generation per chat request, sent in batches to
Langfuse's public ingestion API (`POST {host}/api/public/ingestion`, HTTP Basic public:secret).

HONEST SCOPE. The payload shape comes from Langfuse's public documentation as the author knows it; it is
tested against a mock HTTP server only and has NOT been run against a real Langfuse instance. Like the access
log (R16), it never carries prompt or answer text: only model, alias, client, token counts, timings, cost and
status. A slow or dead Langfuse never touches a request: events go to a bounded queue (full = dropped and
counted) and a background task ships them."""

import asyncio
import base64
import logging
import uuid
from datetime import UTC, datetime

import httpx

log = logging.getLogger("gateway.langfuse")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class LangfuseExporter:
    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        client: httpx.AsyncClient,
        *,
        max_queue: int = 1000,
        batch_size: int = 50,
        flush_s: float = 2.0,
    ) -> None:
        self.url = host.rstrip("/") + "/api/public/ingestion"
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
        self._client = client
        self._queue: asyncio.Queue[dict] = asyncio.Queue(max_queue)
        self._batch_size, self._flush_s = batch_size, flush_s
        self._task: asyncio.Task | None = None
        self.sent = self.dropped = self.errors = 0

    # --- called on the request path: must never block or raise ---------------------------------------

    def record(self, line: dict) -> None:
        """`line` is the access-log dict (no prompt text in it by construction)."""
        try:
            end = line["ts"]
            duration_s = (line.get("duration_ms") or 0) / 1000
            start = end - duration_s
            trace_id = line.get("trace_id") or line["request_id"]
            meta = {
                k: line.get(k)
                for k in (
                    "alias",
                    "client",
                    "provider",
                    "cache",
                    "attempts",
                    "stream_outcome",
                    "error",
                    "status",
                )
            }
            events = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "timestamp": _iso(start),
                    "body": {"id": trace_id, "name": f"chat:{line.get('alias')}", "metadata": meta},
                },
                {
                    "id": str(uuid.uuid4()),
                    "type": "generation-create",
                    "timestamp": _iso(start),
                    "body": {
                        "id": line["request_id"],
                        "traceId": trace_id,
                        "name": line.get("provider") or "-",
                        "model": line.get("model") or None,
                        "startTime": _iso(start),
                        "endTime": _iso(end),
                        "completionStartTime": None
                        if line.get("ttft_ms") is None
                        else _iso(start + line["ttft_ms"] / 1000),
                        "usage": {
                            "input": line.get("tokens_in"),
                            "output": line.get("tokens_out"),
                            "unit": "TOKENS",
                            "totalCost": line.get("cost_usd"),
                        },
                        "level": "ERROR" if (line.get("status") or 200) >= 400 else "DEFAULT",
                        "statusMessage": line.get("error"),
                        "metadata": meta,
                    },
                },
            ]
            for event in events:
                self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
        except Exception:  # noqa: BLE001 - observability must not break a request
            self.errors += 1
            log.exception("langfuse record failed")

    # --- background shipping --------------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.flush()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_s)
            await self.flush()

    async def flush(self) -> None:
        while not self._queue.empty():
            batch = []
            while len(batch) < self._batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            await self._send(batch)

    async def _send(self, batch: list[dict]) -> None:
        try:
            r = await self._client.post(self.url, json={"batch": batch}, headers=self._headers, timeout=5.0)
            if r.status_code >= 300:  # 207 multi-status is fine; real rejections are not
                self.errors += 1
                log.warning("langfuse ingestion answered %s", r.status_code)
            else:
                self.sent += len(batch)
        except httpx.HTTPError as exc:
            self.errors += 1
            log.warning("langfuse unreachable (%s)", type(exc).__name__)

    def snapshot(self) -> dict:
        return {
            "sent": self.sent,
            "dropped": self.dropped,
            "errors": self.errors,
            "queued": self._queue.qsize(),
        }
