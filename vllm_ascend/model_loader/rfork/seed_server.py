import os
import queue
import socket
import threading
import time
from http import HTTPStatus

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from vllm.logger import init_logger

logger = init_logger(__name__)


def start_fastapi_server(port_queue, local_seed_key, info):
    logger.warning("[RFork Seed] Preparing socket with dynamic port...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    _, port = sock.getsockname()
    logger.warning("[RFork Seed] Assigned dynamic port: %s", port)

    app = FastAPI()

    @app.get("/get_rfork_transfer_engine_info")
    def get_rfork_transfer_engine_info(seed_key: str):
        if seed_key == local_seed_key:
            return {"rfork_transfer_engine_info": info}
        return {"rfork_transfer_engine_info": None}

    @app.get("/rfork_fetch_seed")
    def rfork_fetch_seed():
        return {"status": "ok"}

    @app.get("/health_check_with_key")
    def health_check_with_key(seed_key: str):
        if seed_key == local_seed_key:
            return Response(status_code=HTTPStatus.OK)
        return Response(status_code=HTTPStatus.BAD_REQUEST)

    config = uvicorn.Config(app, host=None, port=None, log_level="warning")
    server = uvicorn.Server(config)

    try:
        port_queue.put(port)
    except Exception as e:
        logger.error("[RFork Seed] Failed to send port via queue: %s", e)
        sock.close()
        return

    logger.warning("[RFork Seed] FastAPI server starting on port %s...", port)
    server.run(sockets=[sock])
    sock.close()


def start_rfork_server(local_seed_key, rfork_transfer_engine_info) -> int:
    port_queue = queue.Queue()
    process = threading.Thread(
        target=start_fastapi_server,
        args=(port_queue, local_seed_key, rfork_transfer_engine_info),
        daemon=True,
    )
    process.start()

    try:
        port = port_queue.get(timeout=15)
        if port == -1:
            raise RuntimeError("Child process failed to start server")
    except Exception as e:
        logger.error("[RFork Seed] start server error: %s", e)
        return -1

    health_timeout_sec = float(os.getenv("RFORK_SEED_SERVER_HEALTH_TIMEOUT_SEC", "30"))
    deadline = time.time() + health_timeout_sec
    healthy = False
    while time.time() < deadline:
        time.sleep(0.01)
        url = f"http://127.0.0.1:{port}/health_check_with_key"
        try:
            response = requests.get(
                url,
                params={"seed_key": local_seed_key},
                timeout=10,
            )
            if response.status_code == 200:
                healthy = True
                break
        except Exception as e:
            logger.warning("[RFork Seed] health check failed, retry: %s", e)
    if healthy:
        return port
    logger.error(
        "[RFork Seed] health check timed out after %.1fs for port %s",
        health_timeout_sec,
        port,
    )
    return -1
