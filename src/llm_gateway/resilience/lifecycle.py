"""Orderly shutdown (PLAN §4.9, R23).

SIGTERM -> draining: new requests get 503 and /readyz fails. Active streams get `drain_timeout_s` to finish;
after that `stop` completes and every stream ends with a typed error event, so its resources are released
and its cost can be settled before the platform kills the process."""

import asyncio
import logging
import signal
import threading

log = logging.getLogger(__name__)


class Lifecycle:
    def __init__(self, drain_timeout_s: float = 8.0) -> None:
        self.drain_timeout_s = drain_timeout_s
        self.draining = False
        self.stop: asyncio.Future | None = None  # completes when the drain window closes
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach(self, loop: asyncio.AbstractEventLoop, *, hook_signals: bool = True) -> None:
        self._loop = loop
        self.stop = loop.create_future()
        if hook_signals and threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                self._hook(sig)

    def _hook(self, sig: signal.Signals) -> None:
        previous = signal.getsignal(sig)

        def handler(signum, frame):
            self.begin_drain()
            if callable(previous):
                previous(signum, frame)  # uvicorn's own handler: stop accepting, wait for connections

        signal.signal(sig, handler)

    def begin_drain(self) -> None:
        """Safe from a signal handler or any thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._start)

    def _start(self) -> None:
        if self.draining:
            return
        self.draining = True
        log.warning("draining: active streams have %.1f s to finish", self.drain_timeout_s)
        self._loop.call_later(self.drain_timeout_s, self._close_window)

    def _close_window(self) -> None:
        if self.stop is not None and not self.stop.done():
            self.stop.set_result(None)
