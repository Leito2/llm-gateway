"""M6: robust streaming — keep-alive pings, thinking as liveness, stalls, cancellation on a real server
(PLAN §4, R10, R11)."""

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import uvicorn
from fastapi.testclient import TestClient
from helpers import FULL_KEY, make_app

from llm_gateway.providers.mock import MockProvider

BODY = {"model": "fast", "messages": [{"role": "user", "content": "hello there"}], "stream": True}
HEADERS = {"Authorization": f"Bearer {FULL_KEY}"}


def events(text: str) -> list[str]:
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def content_of(text: str) -> str:
    out = ""
    for e in events(text):
        if e != "[DONE]":
            for choice in json.loads(e).get("choices") or []:
                out += (choice.get("delta") or {}).get("content") or ""
    return out


class Pausing(MockProvider):
    """Real first chunk, then `gap` seconds of silence (or of thinking chunks), then the rest."""

    def __init__(self, gap_s: float, thinking_every_s: float = 0.0) -> None:
        super().__init__()
        self.gap_s, self.thinking_every_s = gap_s, thinking_every_s

    async def stream(self, request, model):
        chunks = [c async for c in MockProvider.stream(self, request, model)]
        yield chunks[0]
        waited = 0.0
        while waited < self.gap_s:
            step = self.thinking_every_s or self.gap_s
            await asyncio.sleep(step)
            waited += step
            if self.thinking_every_s:  # a thinking delta: liveness, no text for the client
                yield {**chunks[0], "choices": []}
        for c in chunks[1:]:
            yield c


def _app(provider, *, ping_s=0.1, stall_s=5.0):
    app = make_app(stream_ping_s=ping_s)
    gw = app.state.gateway
    gw.providers["groq"] = provider
    gw.routes.aliases["fast"].stall_timeout_s = stall_s
    return app


def test_silence_gets_keep_alive_pings_and_the_answer_is_intact():
    app = _app(Pausing(0.45))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", headers=HEADERS, json=BODY)
    assert r.status_code == 200
    assert r.text.count(": ping\n\n") >= 3
    assert "[DONE]" in events(r.text)
    assert content_of(r.text).startswith("Mock reply to:")


def test_pings_are_off_when_the_interval_is_zero():
    app = _app(Pausing(0.3), ping_s=0)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", headers=HEADERS, json=BODY)
    assert ": ping" not in r.text and "[DONE]" in events(r.text)


def test_thinking_chunks_count_as_liveness_so_the_stream_does_not_stall():
    # 0.6 s of thinking, but never 0.15 s of silence: without liveness this would hit stream_stalled.
    app = _app(Pausing(0.6, thinking_every_s=0.05), ping_s=0, stall_s=0.15)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", headers=HEADERS, json=BODY)
    assert "stream_stalled" not in r.text and "[DONE]" in events(r.text)
    assert content_of(r.text).startswith("Mock reply to:")


def test_pings_do_not_hide_a_real_stall():
    app = _app(Pausing(30), ping_s=0.05, stall_s=0.2)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", headers=HEADERS, json=BODY)
    evs = events(r.text)
    assert ": ping" in r.text
    assert json.loads(evs[-1])["error"]["code"] == "stream_stalled" and "[DONE]" not in evs


# --- real server: cancellation and the Go regression ------------------------------------------------


class Watched(MockProvider):
    """Records whether its upstream stream was closed (the client left) and how far it got."""

    def __init__(self, token_delay_s: float) -> None:
        super().__init__(token_delay_s=token_delay_s)
        self.closed = threading.Event()
        self.started = threading.Event()

    async def stream(self, request, model):
        self.started.set()
        try:
            async for chunk in super().stream(request, model):
                yield chunk
        finally:
            self.closed.set()


@contextlib.contextmanager
def serve(app):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)


def test_client_disconnect_closes_the_upstream_and_frees_the_slots():
    watched = Watched(token_delay_s=0.05)
    app = _app(watched, ping_s=10, stall_s=10)
    gw = app.state.gateway
    long_body = {**BODY, "messages": [{"role": "user", "content": "word " * 200}], "max_tokens": 200}
    with serve(app) as base:
        with httpx.Client(base_url=base, timeout=10) as client:
            with client.stream("POST", "/v1/chat/completions", headers=HEADERS, json=long_body) as r:
                lines = r.iter_lines()
                assert next(line for line in lines if line.startswith("data: "))
            # leaving the `with` closes the connection mid-stream
        assert watched.closed.wait(5), "upstream stream was not closed after the client left"
        for _ in range(100):
            if gw.admission.in_flight == 0:
                break
            time.sleep(0.05)
        assert gw.admission.in_flight == 0


def test_stream_outlives_the_handler_on_a_real_server():
    # Go regression: the handler returns before any token exists; the body must still flow.
    watched = Watched(token_delay_s=0.02)
    app = _app(watched)
    with serve(app) as base:
        with httpx.Client(base_url=base, timeout=10) as client:
            r = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert events(r.text)[-1] == "[DONE]"
    assert content_of(r.text) == "Mock reply to: hello there"
    assert watched.closed.is_set()
