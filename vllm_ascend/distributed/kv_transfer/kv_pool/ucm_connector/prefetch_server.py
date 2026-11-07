# SPDX-License-Identifier: Apache-2.0
"""In-process HTTP endpoint for on-demand UCM prefetch.

Listens in the worker process on base_port + rank so the router prefetch
endpoint (/kv/prefetch) can route to the target rank and trigger
CacheStore::Warm (Backend2Cache, SSD -> DDR). Standard library only.

The UCM cache buffer lives in the worker process started by vllm-ascend, so
this server is owned by vllm-ascend rather than the UCM library.
"""

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

from vllm.logger import logger

# UCM BlockId is a 16-byte hash (std::array<std::byte, 16>), i.e. 32 hex chars.
# A mismatched length would be truncated by BufferArrayView (num=0), warming
# nothing while still reporting success, so reject it here instead.
_UCM_BLOCK_ID_BYTES = 16


class _PrefetchHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying the state shared with the request handler."""

    prefetch_fn: Callable[[list[bytes]], None]
    token: str | None
    lock: threading.Lock


class _PrefetchHandler(BaseHTTPRequestHandler):
    server_version = "UcmPrefetchServer/1.0"

    @property
    def _prefetch_server(self) -> _PrefetchHTTPServer:
        return cast(_PrefetchHTTPServer, self.server)

    @property
    def prefetch_fn(self) -> Callable[[list[bytes]], None]:
        return self._prefetch_server.prefetch_fn

    @property
    def token(self) -> str | None:
        return self._prefetch_server.token

    @property
    def lock(self) -> threading.Lock:
        return self._prefetch_server.lock

    def _authorized(self) -> bool:
        if not self.token:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {self.token}"

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    @staticmethod
    def _parse_block_ids(body: dict) -> list[bytes]:
        raw = body.get("block_ids")
        if not isinstance(raw, list):
            raise ValueError("'block_ids' must be an array of hex strings")
        block_ids: list[bytes] = []
        for h in raw:
            try:
                bid = bytes.fromhex(str(h))
            except ValueError:
                raise ValueError(f"invalid hex block id: {h!r}")
            if len(bid) != _UCM_BLOCK_ID_BYTES:
                raise ValueError(
                    f"block id must be {_UCM_BLOCK_ID_BYTES * 2} hex chars ({_UCM_BLOCK_ID_BYTES} bytes), got {h!r}"
                )
            block_ids.append(bid)
        return block_ids

    def do_GET(self) -> None:
        if self.path == "/healthy":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._send_json(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            body = self._read_body()
            if self.path == "/kv/prefetch":
                block_ids = self._parse_block_ids(body)
                with self.lock:
                    self.prefetch_fn(block_ids)
                self._send_json(200, {"ok": True, "count": len(block_ids)})
            else:
                self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as e:  # noqa: BLE001
            logger.warning("UCM prefetch HTTP error: %s", e)
            self._send_json(400, {"ok": False, "error": str(e)})

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("ucm-prefetch-http: %s", fmt % args if args else fmt)


class UcmPrefetchServer:
    def __init__(
        self,
        prefetch_fn: Callable[[list[bytes]], None],
        host: str,
        port: int,
        token: str | None = None,
    ) -> None:
        self.lock = threading.Lock()
        self._server = _PrefetchHTTPServer((host, port), _PrefetchHandler)
        self._server.prefetch_fn = prefetch_fn
        self._server.token = token
        self._server.lock = self.lock
        self._thread = threading.Thread(target=self._server.serve_forever, name="ucm-prefetch-http", daemon=True)

    def start(self) -> "UcmPrefetchServer":
        self._thread.start()
        logger.info("UCM prefetch HTTP started on %s", self._server.server_address)
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
