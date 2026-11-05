# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import errno
import socket
import threading
import time
from contextlib import suppress
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.types import RFORK_PROTOCOL_VERSION, SeedTransferInfo

SERVER_STOP_TIMEOUT_SEC = 5.0
HEALTH_POLL_INTERVAL_SEC = 0.05


class RForkSeedServerHandle:
    """Own the listener and its server thread until both are stopped."""

    def __init__(self, server: uvicorn.Server, sock: socket.socket) -> None:
        self.server = server
        self.sock = sock
        self.port = sock.getsockname()[1]
        self.thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="RForkSeedServer"
        )

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def start(self, timeout: float) -> None:
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("RFork seed server exited during startup")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RFork seed server startup timed out")
            time.sleep(min(HEALTH_POLL_INTERVAL_SEC, remaining))
        if not self.thread.is_alive():
            raise RuntimeError("RFork seed server exited during startup")

    def stop(self, timeout: float = SERVER_STOP_TIMEOUT_SEC) -> bool:
        self.server.should_exit = True
        if self.thread.ident is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=max(float(timeout), 0.0))
        stopped = not self.thread.is_alive()
        if stopped:
            with suppress(OSError):
                self.sock.close()
        else:
            logger.warning("[RFork Seed] server thread did not stop within %.2fs", timeout)
        return stopped


def _create_bound_socket(bind_host: str, port: int = 0) -> socket.socket:
    """Create an IPv4 or IPv6 listener from the bind host's resolver result."""
    host = bind_host or "0.0.0.0"
    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)
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
    raise OSError(f"unable to bind RFork seed host {host!r} port {port}: {last_error}") from last_error


def create_rfork_server(
    seed_key: str,
    info: SeedTransferInfo,
    *,
    bind_host: str = "0.0.0.0",
    port: int = 0,
) -> RForkSeedServerHandle:
    try:
        sock = _create_bound_socket(bind_host, port)
    except OSError as exc:
        bind_error = exc.__cause__ if isinstance(exc.__cause__, OSError) else exc
        if port <= 0 or bind_error.errno != errno.EADDRINUSE:
            raise
        logger.warning("[RFork Seed] configured port %d on %s is in use; using an OS-assigned port", port, bind_host)
        sock = _create_bound_socket(bind_host, 0)

    try:
        app = FastAPI()

        @app.get("/get_rfork_transfer_engine_info")
        def get_rfork_transfer_engine_info(seed_key: str):
            if seed_key != expected_seed_key:
                return Response(status_code=HTTPStatus.NOT_FOUND)
            return {
                "rfork_protocol_version": RFORK_PROTOCOL_VERSION,
                "rfork_transfer_engine_info": (info.session_id, info.weights),
            }

        expected_seed_key = seed_key
        server = uvicorn.Server(uvicorn.Config(app, host=None, port=None, log_level="warning"))
        return RForkSeedServerHandle(server, sock)
    except BaseException:
        sock.close()
        raise
