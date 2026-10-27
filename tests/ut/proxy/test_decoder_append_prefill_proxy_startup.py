# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise real proxy imports and cross-worker state without model backends."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROXY_MODULE = "examples.disaggregated_prefill_v1.decoder_append_prefill_proxy_server"
PROXY_SCRIPT = REPO_ROOT / "examples/disaggregated_prefill_v1/decoder_append_prefill_proxy_server.py"


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def running_proxy(tmp_path, launch_mode: str, workers: int):
    port = unused_port()
    command = [sys.executable]
    command.extend([str(PROXY_SCRIPT)] if launch_mode == "script" else ["-m", PROXY_MODULE])
    command.extend(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--prefiller-ports",
            "19001",
            "--decoder-ports",
            "19002",
            "--workers",
            str(workers),
            "--decoder-data-parallel-size",
            "4",
            "--decoder-engine-epoch",
            "startup-test",
            "--enable-decoder-append-prefill",
        ]
    )
    with (tmp_path / f"{launch_mode}-{workers}.log").open("w+") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=log, start_new_session=True)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=1.0) as client:
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    assert process.poll() is None, log.read()
                    try:
                        if client.get("/healthcheck").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    log.seek(0)
                    pytest.fail(f"Proxy startup timed out:\n{log.read()}")
                yield client, port
        finally:
            # The new process group contains only this fixture's proxy and workers.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


@pytest.mark.parametrize("launch_mode", ["script", "module"])
def test_proxy_entrypoint_starts_with_its_own_scheduler(tmp_path, launch_mode):
    with running_proxy(tmp_path, launch_mode, workers=1) as (client, _):
        health = client.get("/healthcheck").json()
        assert health["decoder_data_parallel_size"] == 4
        assert health["decoder_engine_epoch"] == "startup-test"
        assert health["request_num"] == 0
        assert health["active_sessions"] == 0


def manager_client(port: int, action: str) -> dict:
    # Fresh interpreters exercise Manager registration without sharing module globals.
    code = """
import json
import sys
from examples.disaggregated_prefill_v1 import decoder_append_prefill_proxy_server as proxy

scheduler = proxy.connect_shared_scheduler(int(sys.argv[1]))
action = sys.argv[2]
if action == "bind":
    assert scheduler.reserve_session("shared-session", 60)["status"] == "cold"
    owner = scheduler.reserve_decoder("0.0.0.0:19002", 10, dp_rank=3)
    scheduler.bind_session_decoder("shared-session", owner["key"], owner["dp_rank"])
    scheduler.finish_session("shared-session", True)
    scheduler.release_decoder(owner["key"], 10)
    result = owner
elif action == "reserve":
    result = scheduler.reserve_session("shared-session", 60)
elif action == "finish":
    scheduler.finish_session("shared-session", False)
    result = scheduler.healthcheck()
elif action == "health":
    result = scheduler.healthcheck()
print(json.dumps(result))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(port), action], cwd=REPO_ROOT, text=True, timeout=10
    )
    return json.loads(output)


@pytest.mark.parametrize("launch_mode", ["script", "module"])
def test_multiple_workers_share_session_owner_and_busy_guard(tmp_path, launch_mode):
    with running_proxy(tmp_path, launch_mode, workers=2) as (client, port):
        owner = manager_client(port, "bind")
        warm = manager_client(port, "reserve")
        assert warm["status"] == "warm"
        assert warm["decoder_key"] == owner["key"]
        assert warm["decoder_dp_rank"] == owner["dp_rank"] == 3
        assert warm["decoder_engine_epoch"] == "startup-test"

        # An HTTP worker must see the reservation made by the separate Manager client.
        response = client.post(
            "/v1/completions",
            headers={"X-Session-ID": "shared-session"},
            json={"model": "test", "prompt": "follow-up", "max_tokens": 1},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "session_busy"
        health = manager_client(port, "finish")
        assert health["in_flight_sessions"] == 0
        assert health["request_num"] == 0
        assert health["active_sessions"] == 1

        # Role values sent by a worker must retain their identity in the parent.
        removed = client.post("/instances/remove", json={"type": "decode", "instances": ["127.0.0.1:19002"]})
        assert removed.status_code == 200
        health = manager_client(port, "health")
        assert health["decode_instances"] == 0
        assert health["active_sessions"] == 0
