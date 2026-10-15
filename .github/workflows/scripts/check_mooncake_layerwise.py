# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the installed native Layerwise SDK over TCP before starting the NPU E2E.

This uses CPU buffers, not model KV or NPU transfers. It deliberately imports no
vLLM modules so dependency failures are reported before compiling the plugin.
"""

import argparse
import ctypes
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from importlib import import_module
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

REQUIRED_METHODS = (
    "batch_put_session_start",
    "batch_put_from_multi_buffer_ranges",
    "batch_put_session_end",
    "batch_put_session_revoke",
    "batch_get_session_start",
    "batch_get_into_multi_buffer_ranges",
    "batch_get_session_end",
)
HOST = "127.0.0.1"
MASTER_READY_SECONDS = 30
MASTER_STOP_SECONDS = 5
SEGMENT_BYTES = 64 << 20
KEY_COUNT = 2
LAYER_COUNT = 3
BUFFERS_PER_LAYER = 2
BUFFER_BYTES = 64
LAYER_BYTES = BUFFERS_PER_LAYER * BUFFER_BYTES
OBJECT_BYTES = LAYER_COUNT * LAYER_BYTES
POISON_BYTE = 255


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    arguments = parser.parse_args()
    wheel = distribution("mooncake-transfer-engine-npu")
    require_result("installed version", wheel.version, arguments.expected_version)
    store_module = import_module("mooncake.store")
    cli_module = import_module("mooncake.cli")
    package = Path(str(wheel.locate_file("mooncake"))).resolve()
    # Both imports must resolve to this wheel, not another installation on PYTHONPATH.
    if store_module.__file__ is None or cli_module.__file__ is None:
        raise RuntimeError("Mooncake store and master must resolve to files in the installed wheel")
    require_result("store package", Path(store_module.__file__).resolve().parent, package)
    require_result("master package", Path(cli_module.__file__).resolve().parent, package)
    client_type = store_module.MooncakeDistributedStore
    missing = [name for name in REQUIRED_METHODS if not callable(getattr(client_type, name, None))]
    require_result("missing session/range APIs", missing, [])
    print(f"Mooncake native SDK: version={wheel.version}, package={package}", flush=True)

    with ExitStack() as reservations:
        sockets = [reservations.enter_context(socket.socket()) for _ in range(3)]
        for reserved in sockets:
            reserved.bind((HOST, 0))
        master_port, metrics_port, client_port = [reserved.getsockname()[1] for reserved in sockets]
    # This wheel's CLI execs its bundled master, preserving the PID for cleanup.
    command = [sys.executable, "-m", "mooncake.cli", "--port", str(master_port), "--metrics_port", str(metrics_port)]
    with subprocess.Popen(command) as master:
        try:
            wait_for_master(master, master_port)
            exercise_sessions(store_module, f"{HOST}:{master_port}", f"{HOST}:{client_port}")
        finally:
            master.terminate()
            try:
                master.wait(timeout=MASTER_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait(timeout=MASTER_STOP_SECONDS)
    print("MOONCAKE_LAYERWISE_SDK_SMOKE PASSED: native TCP/CPU ranges; NPU E2E still required", flush=True)


def wait_for_master(master: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + MASTER_READY_SECONDS
    while time.monotonic() < deadline:
        if master.poll() is not None:
            raise RuntimeError(f"Mooncake master exited before becoming ready: {master.returncode}")
        try:
            with socket.create_connection((HOST, port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("Mooncake master did not become ready")


def exercise_sessions(store_module: Any, master_address: str, client_address: str) -> None:
    client = store_module.MooncakeDistributedStore()
    total_bytes = KEY_COUNT * OBJECT_BYTES
    # Distinct patterns per key/layer/buffer expose incorrect range offsets or routing.
    expected = bytes(index // BUFFER_BYTES + 1 for index in range(total_bytes))
    source = (ctypes.c_ubyte * total_bytes).from_buffer_copy(expected)
    destination = (ctypes.c_ubyte * total_bytes)()
    ctypes.memset(ctypes.addressof(destination), POISON_BYTE, total_bytes)
    keys = [f"layerwise-sdk-smoke-{index}" for index in range(KEY_COUNT)]
    try:
        setup = client.setup(client_address, "P2PHANDSHAKE", SEGMENT_BYTES, SEGMENT_BYTES, "tcp", "", master_address)
        require_result("setup", setup, 0)
        for buffer in (source, destination):
            require_result("register_buffer", client.register_buffer(ctypes.addressof(buffer), total_bytes), 0)
        require_result("put start", client.batch_put_session_start(keys, [OBJECT_BYTES] * KEY_COUNT), [0] * KEY_COUNT)
        try:
            # Write in reverse layer order, then read forwards: offsets must select the same bytes.
            for layer in reversed(range(LAYER_COUNT)):
                buffers, sizes, offsets = range_arguments(ctypes.addressof(source), layer)
                result = client.batch_put_from_multi_buffer_ranges(keys, buffers, sizes, offsets)
                require_result(f"put layer {layer}", result, [LAYER_BYTES] * KEY_COUNT)
            require_result("put commit", client.batch_put_session_end(keys), [0] * KEY_COUNT)
        except BaseException:
            client.batch_put_session_revoke(keys)
            raise

        require_result("get start", client.batch_get_session_start(keys), [0] * KEY_COUNT)
        try:
            for layer in range(LAYER_COUNT):
                buffers, sizes, offsets = range_arguments(ctypes.addressof(destination), layer)
                result = client.batch_get_into_multi_buffer_ranges(keys, buffers, sizes, offsets)
                require_result(f"get layer {layer}", result, [LAYER_BYTES] * KEY_COUNT)
        finally:
            require_result("get end", client.batch_get_session_end(keys), 0)
        require_result("byte equality after poisoned destination", bytes(destination), expected)

        revoked = ["layerwise-sdk-smoke-revoke"]
        require_result("revoke start", client.batch_put_session_start(revoked, [OBJECT_BYTES]), [0])
        require_result("put revoke", client.batch_put_session_revoke(revoked), [0])
    finally:
        # Keep the registered buffers alive until the native client releases its references.
        client.close()


def range_arguments(address: int, layer: int) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    offsets = [layer * LAYER_BYTES + index * BUFFER_BYTES for index in range(BUFFERS_PER_LAYER)]
    buffers = [[address + key * OBJECT_BYTES + offset for offset in offsets] for key in range(KEY_COUNT)]
    sizes = [[BUFFER_BYTES] * BUFFERS_PER_LAYER for _ in range(KEY_COUNT)]
    return buffers, sizes, [offsets[:] for _ in range(KEY_COUNT)]


def require_result(operation: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"Mooncake SDK {operation} failed: expected {expected!r}, got {actual!r}")
    print(f"Mooncake SDK {operation}: passed", flush=True)


if __name__ == "__main__":
    main()
