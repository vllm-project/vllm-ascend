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

import queue
import socket
import threading
import time
from http import HTTPStatus

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from vllm.logger import logger


def start_fastapi_server(
    port_queue: queue.Queue[int],
    local_seed_key,
    info,
):
    logger.info("[RFork Seed] Preparing socket with dynamic port...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    _, port = sock.getsockname()
    logger.info("[RFork Seed] Assigned dynamic port: %s", port)

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

    logger.info("[RFork Seed] FastAPI server starting on port %s...", port)
    server.run(sockets=[sock])
    sock.close()


def start_rfork_server(local_seed_key, rfork_transfer_engine_info, health_timeout_sec: float = 30.0) -> int:
    port_queue: queue.Queue[int] = queue.Queue()
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

    deadline = time.time() + health_timeout_sec
    healthy = False
    retry_count = 0
    last_error = None
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
            last_error = f"unexpected status code {response.status_code} from health check"
        except Exception as e:
            last_error = str(e)
        retry_count += 1
    if healthy:
        if retry_count > 1:
            logger.info(
                "[RFork Seed] health check passed after %d retries for port %s",
                retry_count - 1,
                port,
            )
        return port
    logger.error(
        "[RFork Seed] health check timed out after %.1fs for port %s, last error: %s",
        health_timeout_sec,
        port,
        last_error,
    )
    return -1
