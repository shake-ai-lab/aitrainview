"""Thread-safe pub/sub that fans worker-thread events out to WebSocket clients.

Binary frames: [u32 LE header length][JSON header (space padded to 4 bytes)][payload]
"""
from __future__ import annotations

import asyncio
import json
import math
import struct


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return _clean(o.item())
        except Exception:
            return str(o)
    return o


def dumps(o) -> str:
    try:
        return json.dumps(o, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError):
        return json.dumps(_clean(o), ensure_ascii=False, allow_nan=False, default=str)


def frame(header: dict, payload: bytes = b"") -> bytes:
    h = dumps(header).encode("utf-8")
    h += b" " * ((-(4 + len(h))) % 4)
    return struct.pack("<I", len(h)) + h + payload


MAX_BACKLOG = 4000  # per client; beyond this, streaming telemetry is dropped for that client
# state changes the UI must never miss (they are rare, so they may exceed the backlog)
CRITICAL = {"runtime", "train_start", "train_done", "train_error", "train_reset", "infer_start", "infer_done",
            "infer_error", "atlas_done", "atlas_error", "status"}


class Bus:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._subs: set[asyncio.Queue] = set()
        self._dropping: set[int] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)
        self._dropping.discard(id(q))

    def publish(self, msg, critical: bool | None = None) -> None:
        if critical is None:
            critical = isinstance(msg, dict) and msg.get("type") in CRITICAL
        data = msg if isinstance(msg, (bytes, bytearray, str)) else dumps(msg)
        loop = self.loop
        if loop is None or loop.is_closed():
            return

        def put() -> None:
            for q in list(self._subs):
                if critical or q.qsize() < MAX_BACKLOG:
                    q.put_nowait(data)
                    if q.qsize() < MAX_BACKLOG // 2:
                        self._dropping.discard(id(q))
                elif id(q) not in self._dropping:
                    self._dropping.add(id(q))
                    print(f"[events] a client is {MAX_BACKLOG} messages behind; dropping telemetry for it", flush=True)

        try:
            loop.call_soon_threadsafe(put)
        except RuntimeError:
            pass

    def publish_frame(self, header: dict, payload: bytes = b"") -> None:
        self.publish(frame(header, payload))

    def status(self, message: str, level: str = "info") -> None:
        self.publish({"type": "status", "message": message, "level": level})
