#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import math
import queue
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.types import SeedTransferInfo

SERVER_STARTUP_QUEUE_TIMEOUT_SEC = 15.0
SERVER_STOP_TIMEOUT_SEC = 5.0
HEALTH_POLL_INTERVAL_SEC = 0.01


@dataclass(slots=True)
class _ServerStartup:
    server: uvicorn.Server
    sock: socket.socket
    port: int


class RForkSeedServerHandle:
    """Own the RFork HTTP server and prevent failed health checks from leaving zombie threads."""

    def __init__(
        self,
        *,
        server: uvicorn.Server,
        sock: socket.socket,
        thread: threading.Thread,
        port: int,
    ) -> None:
        self.server = server
        self.sock = sock
        self.thread = thread
        self.port = port
        self._stop_lock = threading.Lock()
        self._stopped = False

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def stop(self, timeout: float = SERVER_STOP_TIMEOUT_SEC) -> bool:
        """Request shutdown and join the server thread, idempotently."""
        with self._stop_lock:
            if not self._stopped:
                self.server.should_exit = True
                # Closing the socket wakes uvicorn versions waiting for socket activity.
                with suppress(OSError):
                    self.sock.close()
                self._stopped = True

        if self.thread is not threading.current_thread():
            self.thread.join(timeout=max(float(timeout), 0.0))
        stopped = not self.thread.is_alive()
        if not stopped:
            logger.warning("[RFork Seed] server thread did not stop within %.2fs", timeout)
        return stopped


def _create_bound_socket(bind_host: str) -> socket.socket:
    """Create an IPv4 or IPv6 listener from the bind host's resolver result."""
    host = bind_host or "0.0.0.0"
    infos = socket.getaddrinfo(host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)
    last_error: OSError | None = None
    for family, socktype, protocol, _, sockaddr in infos:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(family, socktype, protocol)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            sock.listen(socket.SOMAXCONN)
            return sock
        except OSError as exc:
            last_error = exc
            if sock is not None:
                sock.close()
    if last_error is None:
        raise OSError(f"unable to resolve RFork seed bind host {host!r}")
    raise OSError(f"unable to bind RFork seed host {host!r}: {last_error}") from last_error


def _health_url(bind_host: str, sock: socket.socket, port: int) -> str:
    if sock.family == socket.AF_INET6:
        # Use the resolved address because IPv6 brackets require a literal rather than a hostname.
        host = str(sock.getsockname()[0]) or "::1"
        if host == "::" or host == "0:0:0:0:0:0:0:0":
            host = "::1"
        return f"http://[{host}]:{port}/health_check_with_key"
    host = str(sock.getsockname()[0]) or bind_host or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{port}/health_check_with_key"


def start_fastapi_server(
    port_queue: queue.Queue[Any],
    expected_seed_key,
    info: SeedTransferInfo,
    *,
    bind_host: str = "0.0.0.0",
    stop_event: threading.Event | None = None,
):
    logger.debug("[RFork Seed] Preparing socket with dynamic port on %s...", bind_host)

    sock: socket.socket | None = None
    try:
        sock = _create_bound_socket(bind_host)
        _, port = sock.getsockname()[:2]
        logger.debug("[RFork Seed] Assigned dynamic port: %s", port)

        app = FastAPI()
        rfork_transfer_engine_info = (info.session_id, info.weights)
        rfork_transfer_engine_shape_info = info.shapes

        @app.get("/get_rfork_transfer_engine_info")
        def get_rfork_transfer_engine_info(seed_key: str):
            if seed_key == expected_seed_key:
                return {
                    "rfork_transfer_engine_info": rfork_transfer_engine_info,
                    "rfork_transfer_engine_shape_info": rfork_transfer_engine_shape_info,
                }
            return {
                "rfork_transfer_engine_info": None,
                "rfork_transfer_engine_shape_info": None,
            }

        @app.get("/get_rfork_transfer_engine_shape_info")
        def get_rfork_transfer_engine_shape_info(seed_key: str):
            if seed_key == expected_seed_key:
                return {"rfork_transfer_engine_shape_info": rfork_transfer_engine_shape_info}
            return {"rfork_transfer_engine_shape_info": None}

        @app.get("/rfork_fetch_seed")
        def rfork_fetch_seed():
            return {"status": "ok"}

        @app.get("/health_check_with_key")
        def health_check_with_key(seed_key: str):
            if seed_key == expected_seed_key:
                return Response(status_code=HTTPStatus.OK)
            return Response(status_code=HTTPStatus.BAD_REQUEST)

        config = uvicorn.Config(app, host=None, port=None, log_level="warning")
        server = uvicorn.Server(config)
        startup = _ServerStartup(server=server, sock=sock, port=port)
        try:
            port_queue.put(startup)
        except Exception as exc:
            logger.error("[RFork Seed] Failed to send server handle via queue: %s", exc)
            server.should_exit = True
            sock.close()
            return

        if stop_event is not None and stop_event.is_set():
            server.should_exit = True
            sock.close()
            return

        logger.debug("[RFork Seed] FastAPI server starting on port %s...", port)
        server.run(sockets=[sock])
    except Exception as exc:
        logger.error("[RFork Seed] server thread failed: %s", exc)
        with suppress(Exception):
            port_queue.put(exc)
    finally:
        if sock is not None:
            with suppress(OSError):
                sock.close()


def _stop_startup_thread(
    thread: threading.Thread,
    startup_state: dict[str, _ServerStartup],
    stop_event: threading.Event,
    timeout: float = SERVER_STOP_TIMEOUT_SEC,
) -> bool:
    stop_event.set()
    startup = startup_state.get("startup")
    if startup is not None:
        startup.server.should_exit = True
        with suppress(OSError):
            startup.sock.close()
    thread.join(timeout=max(float(timeout), 0.0))
    if thread.is_alive():
        logger.warning("[RFork Seed] failed to join startup thread within %.2fs", timeout)
        return False
    return True


def start_rfork_server(
    seed_key,
    rfork_transfer_engine_info: SeedTransferInfo,
    health_timeout_sec: float = 30.0,
    *,
    bind_host: str = "0.0.0.0",
) -> RForkSeedServerHandle:
    if isinstance(health_timeout_sec, bool) or not isinstance(health_timeout_sec, (int, float)):
        raise ValueError("health_timeout_sec must be a finite positive number")
    if not math.isfinite(float(health_timeout_sec)) or float(health_timeout_sec) <= 0:
        raise ValueError("health_timeout_sec must be a finite positive number")

    port_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    startup_stop_event = threading.Event()
    startup_state: dict[str, _ServerStartup] = {}

    def run_server() -> None:
        start_fastapi_server(
            port_queue,
            seed_key,
            rfork_transfer_engine_info,
            bind_host=bind_host,
            stop_event=startup_stop_event,
        )

    thread = threading.Thread(target=run_server, daemon=True, name="RForkSeedServer")
    thread.start()
    deadline = time.monotonic() + float(health_timeout_sec)

    try:
        remaining = deadline - time.monotonic()
        startup = port_queue.get(timeout=min(SERVER_STARTUP_QUEUE_TIMEOUT_SEC, max(remaining, 0.0)))
        if isinstance(startup, BaseException):
            raise RuntimeError("Child thread failed to start server") from startup
        if not isinstance(startup, _ServerStartup):
            raise RuntimeError(f"Child thread returned invalid server startup state: {startup!r}")
        startup_state["startup"] = startup
    except Exception as exc:
        logger.error("[RFork Seed] start server error for seed_key=%s: %s", seed_key, exc)
        _stop_startup_thread(thread, startup_state, startup_stop_event)
        raise RuntimeError("RFork seed server failed to start") from exc

    handle = RForkSeedServerHandle(
        server=startup.server,
        sock=startup.sock,
        thread=thread,
        port=startup.port,
    )
    url = _health_url(bind_host, startup.sock, startup.port)
    healthy = False
    retry_count = 0
    last_error: object | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            response = requests.get(
                url,
                params={"seed_key": seed_key},
                timeout=min(10.0, remaining),
            )
            if response.status_code == HTTPStatus.OK:
                healthy = True
                break
            last_error = f"unexpected status code {response.status_code} from health check"
        except Exception as exc:
            last_error = str(exc)
        retry_count += 1
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(HEALTH_POLL_INTERVAL_SEC, remaining))

    if healthy:
        if retry_count > 1:
            logger.info(
                "[RFork Seed] health check passed after %d retries for port %s",
                retry_count - 1,
                startup.port,
            )
        return handle

    logger.error(
        "[RFork Seed] health check timed out after %.1fs for port %s, last error: %s",
        health_timeout_sec,
        startup.port,
        last_error,
    )
    # Give cleanup a separate bounded budget so health timeout cannot leave a live thread.
    handle.stop()
    raise RuntimeError(f"RFork seed server health check failed: {last_error}")
