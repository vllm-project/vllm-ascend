"""Exercise listener selection and Uvicorn's worker lifecycle without a model."""

import contextlib
import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from examples.disaggregated_prefill_v1 import proxy_worker_listener as listener

PROXY_DIR = Path(__file__).resolve().parents[3] / "examples/disaggregated_prefill_v1"


@pytest.mark.parametrize("shared_listener", [False, True], ids=["auto", "shared-fallback"])
def test_worker_connections_restart_and_shutdown(tmp_path, shared_listener):
    script = tmp_path / "probe.py"
    script.write_text(
        "import os, sys, signal, threading, time\n"
        "from pathlib import Path\n"
        "from fastapi import FastAPI\n"
        "import proxy_worker_listener as listener\n"
        "app = FastAPI()\n"
        "@app.get('/')\n"
        "def pid(): return {'pid': os.getpid()}\n"
        "def create_app(): return app\n"
        "def stop_on_file():\n"
        "    while not Path(sys.argv[2]).exists(): time.sleep(0.05)\n"
        "    signal.raise_signal(signal.SIGTERM)\n"
        "if __name__ == '__main__':\n"
        "    threading.Thread(target=stop_on_file, daemon=True).start()\n"
        "    if sys.argv[3] == 'shared': listener.supports_worker_listeners = lambda host: False\n"
        "    listener.run_worker_listeners('probe:create_app', host='127.0.0.1', port=int(sys.argv[1]), workers=4)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROXY_DIR), os.environ.get("PYTHONPATH", "")]))
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    stop_file = tmp_path / "stop"
    command = [sys.executable, str(script), str(port), str(stop_file), "shared" if shared_listener else "auto"]

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
        proc = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        children = []
        try:
            deadline = time.monotonic() + 30
            workers = set()
            while len(workers) != 4 and time.monotonic() < deadline:
                assert proc.poll() is None, log_path.read_text()
                with contextlib.suppress(OSError, http.client.HTTPException):
                    workers.add(request_pid())
                time.sleep(0.05)
            assert len(workers) == 4, log_path.read_text()
            served_workers = {request_pid() for _ in range(200)}
            assert served_workers <= workers
            if not shared_listener and listener.supports_worker_listeners("127.0.0.1"):
                assert served_workers == workers
            killed = next(iter(workers))
            psutil.Process(killed).kill()
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
            children = psutil.Process(proc.pid).children(recursive=True)
            stop_file.touch()
            proc.wait(timeout=15)
            assert proc.returncode == 0, log_path.read_text()
            with pytest.raises(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
        finally:
            if proc.poll() is None:
                children = psutil.Process(proc.pid).children(recursive=True)
                proc.kill()
                proc.wait(timeout=15)
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
            psutil.wait_procs(children, timeout=5)

    # An occupied port must fail before workers join an unrelated reuse group.
    stop_file.unlink()
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if listener.supports_worker_listeners("127.0.0.1"):
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        occupied.bind(("127.0.0.1", port))
        occupied.listen()
        result = subprocess.run(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert result.returncode != 0
