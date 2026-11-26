"""Exercise real Linux listeners and Uvicorn's worker lifecycle without a model."""

import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="SO_REUSEPORT listener behavior is Linux-specific")
PROXY_DIR = Path(__file__).resolve().parents[3] / "examples/disaggregated_prefill_v1"


def test_independent_listeners_distribute_connections_and_restart(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(
        "import os, sys\n"
        "from fastapi import FastAPI\n"
        "from proxy_worker_listener import run_worker_listeners\n"
        "app = FastAPI()\n"
        "@app.get('/')\n"
        "def pid(): return {'pid': os.getpid()}\n"
        "def create_app(): return app\n"
        "if __name__ == '__main__':\n"
        "    run_worker_listeners('probe:create_app', host='127.0.0.1', port=int(sys.argv[1]), workers=4)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROXY_DIR), os.environ.get("PYTHONPATH", "")]))
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    command = [sys.executable, str(script), str(port)]

    def listener_count():
        return sum(
            fields[1].endswith(f":{port:04X}") and fields[3] == "0A"
            for line in Path("/proc/net/tcp").read_text().splitlines()[1:]
            if (fields := line.split())
        )

    def request_pid():
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            conn.request("GET", "/", headers={"Connection": "close"})
            response = conn.getresponse()
            assert response.status == 200
            return json.loads(response.read())["pid"]
        finally:
            conn.close()

    log_path = tmp_path / "workers.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while listener_count() != 4 and time.monotonic() < deadline:
                assert proc.poll() is None, log_path.read_text()
                time.sleep(0.1)
            assert listener_count() == 4, log_path.read_text()
            workers = {request_pid() for _ in range(200)}
            assert len(workers) == 4
            killed = next(iter(workers))
            os.kill(killed, signal.SIGKILL)
            deadline = time.monotonic() + 30
            replacement = None
            while time.monotonic() < deadline:
                with contextlib.suppress(OSError, http.client.HTTPException):
                    pid = request_pid()
                    if pid not in workers:
                        replacement = pid
                        break
                time.sleep(0.1)
            assert replacement is not None, log_path.read_text()
            assert listener_count() == 4
            proc.terminate()
            proc.wait(timeout=15)
            assert listener_count() == 0
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)

    # An occupied port must fail before workers join an unrelated reuse group.
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        occupied.bind(("127.0.0.1", port))
        occupied.listen()
        result = subprocess.run(command, env=env, capture_output=True, timeout=15)
        assert result.returncode != 0
        assert b"Address already in use" in result.stderr
