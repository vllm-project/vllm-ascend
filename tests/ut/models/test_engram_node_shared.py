# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from unittest.mock import Mock

import pytest

from vllm_ascend import engram_node_shared as shared

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Engram sharing uses Linux memfds and peer credentials")

TABLE = "layers.0.engram.embed.weight"
SCALE = "layers.0.engram.embed.scale"
GENERATION = "test-generation"
CLI = (sys.executable, "-m", "vllm_ascend.engram_node_shared")


@pytest.fixture
def checkpoint(tmp_path):
    shard = "model.safetensors"
    config = {"engram_layer_ids": [0], "engram_num_embeddings": [1], "engram_head_dim": 32}
    index = {"weight_map": {TABLE: shard, SCALE: shard}}
    header = {
        TABLE: {"dtype": "F8_E4M3", "shape": [1, 32], "data_offsets": [0, 32]},
        SCALE: {"dtype": "F8_E8M0", "shape": [1, 1], "data_offsets": [32, 33]},
    }
    header_bytes = json.dumps(header).encode()
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / shard).write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(range(33)))
    return tmp_path


def test_owner_help():
    result = subprocess.run([*CLI, "--help"], check=True, capture_output=True, text=True, timeout=30)
    assert "serve,status,wait,drain" in result.stdout


@pytest.mark.parametrize("fail_load", [False, True])
def test_load_publishes_only_complete_tables(checkpoint, monkeypatch, fail_load):
    owner = shared.NodeTableOwner(str(checkpoint / "owner.sock"), GENERATION, checkpoint)
    load_memfd = shared._load_sealed_memfd
    reached_scales = threading.Event()
    resume_loading = threading.Event()
    opened: list[int] = []

    def load_one(part):
        if part["name"] == SCALE:
            reached_scales.set()
            assert resume_loading.wait(timeout=10)
            if fail_load:
                raise OSError("scale load failed")
        fd = load_memfd(part)
        opened.append(fd)
        return fd

    monkeypatch.setattr(shared, "_load_sealed_memfd", load_one)
    loader = threading.Thread(target=owner.load)
    loader.start()
    try:
        assert reached_scales.wait(timeout=10)
        with owner.lock:
            status = owner.status()
        assert status["state"] == "loading"
        assert status["backing_inodes"] == {}
    finally:
        resume_loading.set()
        loader.join(timeout=10)
    assert not loader.is_alive()
    if fail_load:
        assert owner.state == "failed"
        assert owner.handles == {}
        for fd in opened:
            with pytest.raises(OSError):
                os.fstat(fd)
    else:
        try:
            assert owner.state == "ready"
            assert owner.handles == {TABLE: opened}
            assert len(owner.status()["backing_inodes"][TABLE]) == 2
        finally:
            for fd in opened:
                os.close(fd)


def test_failed_fd_delivery_drops_lease(checkpoint):
    owner = shared.NodeTableOwner(str(checkpoint / "owner.sock"), GENERATION, checkpoint)
    owner.load()
    client, server = socket.socketpair()
    connection = Mock(wraps=server)
    connection.sendmsg.side_effect = BrokenPipeError("FD delivery failed")
    try:
        with client, server:
            client.settimeout(2)
            shared._send_json(
                client,
                {
                    "op": "attach",
                    "version": shared.PROTOCOL_VERSION,
                    "generation": GENERATION,
                    "fingerprint": owner.layout["fingerprint"],
                    "key": TABLE,
                },
            )
            owner.handle(connection)
            assert not owner.leases
            assert shared._recv_json(client)["ok"]
            assert not shared._recv_json(client)["ok"]
    finally:
        for fd in owner.handles[TABLE]:
            os.close(fd)


@pytest.mark.parametrize("close_peer", [False, True])
def test_release_closes_descriptors_when_exchange_fails(close_peer):
    client, server = socket.socketpair()
    with client, server:
        fds = os.pipe()
        lease = shared.NodeTableLease.__new__(shared.NodeTableLease)
        lease.connection = client
        lease.fds = list(fds)
        lease.metadata = {"lease": "test"}
        if close_peer:
            server.close()
        else:
            server.shutdown(socket.SHUT_WR)
        with pytest.raises((OSError, EOFError)):
            lease.release()
        assert client.fileno() == -1
        assert lease.fds == []
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        lease.release()


def test_drain_waits_for_disconnected_worker_to_exit(checkpoint):
    try:
        os.close(os.pidfd_open(os.getpid()))
    except OSError as exc:
        pytest.skip(f"pidfd_open unavailable: {exc}")
    endpoint = str(checkpoint / "owner.sock")
    with subprocess.Popen(
        [*CLI, "serve", "--model", str(checkpoint), "--endpoint", endpoint, "--generation", GENERATION],
        stdout=subprocess.DEVNULL,
    ) as owner:
        try:
            ready = subprocess.run(
                [*CLI, "wait", "--endpoint", endpoint, "--timeout", "10"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            status = json.loads(ready.stdout)
            fingerprint = status["fingerprint"]
            lease = shared.NodeTableLease(endpoint, GENERATION, fingerprint, TABLE, timeout=2)
            try:
                assert lease.metadata["inodes"] == status["backing_inodes"][TABLE]
                assert not any(os.get_inheritable(fd) for fd in lease.fds)
                assert [os.pread(fd, 1, 0) for fd in lease.fds] == [b"\x00", b"\x20"]
            finally:
                lease.release()
            assert shared.request(endpoint, {"op": "status"})["active_leases"] == 0

            worker_code = """
import sys
from vllm_ascend.engram_node_shared import NodeTableLease

lease = NodeTableLease(*sys.argv[1:], timeout=2)
lease.connection.close()
sys.stdin.read()
"""
            with subprocess.Popen(
                [sys.executable, "-c", worker_code, endpoint, GENERATION, fingerprint, TABLE], stdin=subprocess.PIPE
            ) as worker:
                try:
                    deadline = time.monotonic() + 10
                    while True:
                        status = shared.request(endpoint, {"op": "status"}, timeout=2)
                        if status["stale_leases"] == 1:
                            break
                        assert worker.poll() is None and time.monotonic() < deadline
                        time.sleep(0.05)
                    draining = shared.request(endpoint, {"op": "drain"}, timeout=2)
                    assert draining["state"] == "draining"
                    assert draining["stale_leases"] == 1
                    with pytest.raises(subprocess.TimeoutExpired):
                        owner.wait(timeout=1.2)
                finally:
                    worker.kill()
                    worker.wait(timeout=5)
            assert owner.wait(timeout=5) == 0
        finally:
            if owner.poll() is None:
                owner.kill()
