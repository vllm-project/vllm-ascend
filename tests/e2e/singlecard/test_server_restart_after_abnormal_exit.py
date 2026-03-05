#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Test server restart after abnormal exit with PDEATHSIG guard.

Verifies that:
1. Server can be killed abnormally (SIGKILL or SIGTERM)
2. NPU resources are properly released (verified by successful restart)
3. Server can be restarted and serve requests normally

VLLM_ASCEND_ENABLE_PDEATHSIG_GUARD is enabled by default, so this test
validates the out-of-the-box behaviour.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


MODEL_PATH = os.getenv("VLLM_TEST_MODEL_PATH", "")
HOST = "127.0.0.1"
PORT = 8199


def _wait_port_closed(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            sock = socket.socket()
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            time.sleep(0.5)
        except (OSError, socket.error):
            return True
    return False


def _wait_port_open(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            sock = socket.socket()
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            return True
        except (OSError, socket.error):
            time.sleep(1)
    return False


def _start_server(env_overrides: dict | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_ASCEND_ENABLE_PDEATHSIG_GUARD"] = "1"
    env["VLLM_PLUGINS"] = "ascend"
    if env_overrides:
        env.update(env_overrides)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--max-num-seqs", "32",
        "--served-model-name", "test_model",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.8",
        "--enforce-eager",
    ]
    return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)


def _find_descendant_pids(root_pid: int) -> list[int]:
    """Return all descendant PIDs of the given process (recursive)."""
    try:
        parent = psutil.Process(root_pid)
        return [c.pid for c in parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


@pytest.mark.skipif(
    not MODEL_PATH or not Path(MODEL_PATH).exists(),
    reason="Set VLLM_TEST_MODEL_PATH to a valid local model directory",
)
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PR_SET_PDEATHSIG is only supported on Linux",
)
@pytest.mark.parametrize("kill_signal", [signal.SIGKILL, signal.SIGTERM])
def test_server_restart_after_abnormal_exit(kill_signal: int):
    """Kill -> verify cleanup -> restart -> verify healthy."""

    if _wait_port_open(HOST, PORT, timeout_s=1.0):
        pytest.skip(f"Port {PORT} is already in use")

    # --- Round 1: start, verify, kill ---
    proc = _start_server()
    try:
        assert _wait_port_open(HOST, PORT, timeout_s=300), \
            "Server did not start in time"

        child_pids = _find_descendant_pids(proc.pid)
        all_pids = [proc.pid] + child_pids

        signal_name = signal.Signals(kill_signal).name
        print(f"\n[Test] Killing server (pid={proc.pid}) with {signal_name}, "
              f"known descendants: {child_pids}")

        os.kill(proc.pid, kill_signal)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # Wait for port to close and all descendants to exit
    assert _wait_port_closed(HOST, PORT, timeout_s=30), \
        f"Port {PORT} should be closed after server death"

    # Give the kernel a moment to deliver SIGKILL via pdeathsig
    time.sleep(3)

    still_alive = [p for p in all_pids if psutil.pid_exists(p)]
    if still_alive:
        print(f"[WARN] Descendants still alive after kill: {still_alive}")

    # Extra buffer for NPU driver cleanup
    time.sleep(5)

    # --- Round 2: restart and verify ---
    print("\n[Test] Restarting server …")
    proc2 = _start_server()
    try:
        assert _wait_port_open(HOST, PORT, timeout_s=300), \
            "Server did not restart – NPU resources may not have been released"
        print(f"\n[Test] PASS – Server restarted after {signal.Signals(kill_signal).name}")
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc2.kill()
            proc2.wait()
        _wait_port_closed(HOST, PORT, timeout_s=15)


@pytest.mark.skipif(
    not MODEL_PATH or not Path(MODEL_PATH).exists(),
    reason="Set VLLM_TEST_MODEL_PATH to a valid local model directory",
)
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PR_SET_PDEATHSIG is only supported on Linux",
)
def test_no_orphan_processes_after_sigkill():
    """After SIGKILL on the main process, no descendant should survive."""

    if _wait_port_open(HOST, PORT, timeout_s=1.0):
        pytest.skip(f"Port {PORT} is already in use")

    proc = _start_server()
    try:
        assert _wait_port_open(HOST, PORT, timeout_s=300), \
            "Server did not start in time"

        child_pids = _find_descendant_pids(proc.pid)
        print(f"\n[Test] Server pid={proc.pid}, descendants={child_pids}")

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Allow pdeathsig + atexit cleanup to propagate
    time.sleep(8)

    survivors = [pid for pid in child_pids if psutil.pid_exists(pid)]
    if survivors:
        print(f"[FAIL] Orphan processes detected: {survivors}")
        for pid in survivors:
            try:
                p = psutil.Process(pid)
                print(f"  pid={pid} name={p.name()} status={p.status()}")
                os.kill(pid, signal.SIGKILL)
            except (psutil.NoSuchProcess, OSError):
                pass

    assert not survivors, (
        f"Orphan processes found after SIGKILL: {survivors}. "
        "PDEATHSIG guard or atexit cleanup may have failed."
    )
