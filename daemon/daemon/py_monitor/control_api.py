from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional


@dataclass
class ControlCommand:
    action: str
    payload: Dict[str, Any]
    response_queue: "queue.Queue[Dict[str, Any]]"


class ControlApiServer:
    def __init__(
        self,
        host: str,
        port: int,
        command_queue: "queue.Queue[ControlCommand]",
        snapshot_provider: Callable[[], list],
    ) -> None:
        self._host = host
        self._port = port
        self._command_queue = command_queue
        self._snapshot_provider = snapshot_provider
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return

        command_queue = self._command_queue
        snapshot_provider = self._snapshot_provider

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code: int, body: Dict[str, Any]) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("empty request body")
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("request body must be a JSON object")
                return data

            def _enqueue(self, action: str, payload: Dict[str, Any]) -> None:
                resp_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
                command_queue.put(ControlCommand(action=action, payload=payload, response_queue=resp_q))
                try:
                    result = resp_q.get(timeout=5.0)
                except queue.Empty:
                    self._json(504, {"ok": False, "error": "command timeout"})
                    return
                status = int(result.get("status", 200))
                self._json(status, result)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self._json(200, {"ok": True, "ts": int(time.time())})
                    return
                if self.path == "/v1/pools":
                    pools = []
                    for snap in snapshot_provider():
                        pools.append(
                            {
                                "device_id": int(snap.device_id),
                                "granularity": int(snap.granularity),
                                "total_handles": int(snap.total_handles),
                                "used_handles": int(snap.used_handles),
                                "available_handles": int(snap.available_handles),
                                "total_bytes": int(snap.total_bytes),
                                "used_bytes": int(snap.used_bytes),
                                "available_bytes": int(snap.available_bytes),
                            }
                        )
                    self._json(200, {"ok": True, "pools": pools})
                    return
                self._json(404, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    payload = self._read_json()
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return

                if self.path == "/v1/pools/create":
                    self._enqueue("create", payload)
                    return
                if self.path == "/v1/pools/extend":
                    self._enqueue("extend", payload)
                    return
                if self.path == "/v1/pools/remove":
                    self._enqueue("remove", payload)
                    return

                self._json(404, {"ok": False, "error": "not found"})

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


def parse_int_field(payload: Dict[str, Any], key: str, minimum: int = 0) -> int:
    if key not in payload:
        raise ValueError(f"missing field: {key}")
    value = int(payload[key])
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value
