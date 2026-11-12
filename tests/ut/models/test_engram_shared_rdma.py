# SPDX-License-Identifier: Apache-2.0
"""Protocol and packed-row checks, including the real native packer when built."""

import argparse
import ast
import ctypes
import importlib.util
import json
import os
import signal
import socket
import struct
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def rdma():
    package_spec = importlib.util.find_spec("vllm_ascend")
    assert package_spec is not None and package_spec.origin is not None
    package = Path(package_spec.origin).parent
    source = Path(os.getenv("ENGRAM_TEST_SOURCE", package / "models/deepseek_v41/engram/shared_rdma.py"))
    spec = importlib.util.spec_from_file_location("engram_shared_rdma_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", [None, "a2", "a3", "a5"])
def test_shared_engram_entry_dispatches_to_owner(rdma, monkeypatch, backend):
    calls = []
    monkeypatch.setitem(sys.modules, "vllm_ascend", ModuleType("vllm_ascend"))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.shared_engram_rdma", SimpleNamespace(main=lambda: calls.append("rdma"))
    )
    entry = Path(rdma.__file__).resolve().parents[3] / "shared_engram.py"
    module = ast.parse(entry.read_text())
    dispatch = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    namespace = {"argparse": argparse, "main": lambda: calls.append("a3")}
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(entry), "exec"), namespace)
    argv = ["owner", "--help"] if backend is None else ["owner", "--backend", backend]
    monkeypatch.setattr(sys, "argv", argv)
    namespace["dispatch"]()
    assert calls == ["a3" if backend in (None, "a3") else "rdma"]


@pytest.mark.parametrize("remote", [False, True])
def test_gathered_layer_slice_preserves_row_stride(rdma, remote):
    source = Path(rdma.__file__).parent / "embedding.py"
    module = ast.parse(source.read_text())
    cls = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AscendParallelEngramEmbedding"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "embed_gathered")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    indices = torch.arange(24).reshape(4, 2, 3)[:, 1, :]
    assert not indices.is_contiguous()
    cpu_ids = indices.clone() if remote else None

    def lookup(received, out, **kwargs):
        assert received.data_ptr() == indices.data_ptr()
        assert received.stride() == indices.stride()
        assert kwargs.get("cpu_ids") is cpu_ids
        out.copy_(received.unsqueeze(-1))

    embedding = SimpleNamespace(part_n_hash_cols=3, dim=1, n_hash_cols=3, dp_size=1, tp_size=1, lookup=lookup)
    actual = namespace["embed_gathered"](embedding, indices, 4, cpu_ids=cpu_ids)
    torch.testing.assert_close(actual.float(), indices.unsqueeze(-1).float())


def test_control_round_trip_and_closed_connection(rdma):
    left, right = socket.socketpair()
    try:
        rdma._send(left, {"generation": "g", "count": 17})
        assert rdma._receive(right) == {"generation": "g", "count": 17}
        left.close()
        with pytest.raises(ConnectionError):
            rdma._receive(right)
    finally:
        left.close()
        right.close()


def test_oversized_control_is_rejected_before_payload(rdma):
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", rdma.MAX_CONTROL_BYTES + 1))
        with pytest.raises(ValueError, match="message size"):
            rdma._receive(right)
    finally:
        left.close()
        right.close()


def test_hixl_engine_passes_explicit_options_to_native(rdma, monkeypatch):
    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            if self.name == "engram_hixl_create_v2":
                calls.append(args[:4])
            return 0

    class Library:
        def __getattr__(self, name):
            function = Function(name)
            setattr(self, name, function)
            return function

    monkeypatch.setattr(rdma.ctypes, "CDLL", lambda path: Library())
    rdma.HixlEngine("owner:123", 2, "fake.so", listen_port=26666)
    rdma.HixlEngine("reader:123", 1, "fake.so", local_comm_res="/tmp/ub-{physical}.json")
    assert calls == [
        (b"owner:123", 2, b"26666", None),
        (b"reader:123", 1, None, b"/tmp/ub-{physical}.json"),
    ]


def test_reader_sends_physical_device_id(rdma, monkeypatch):
    client, server = socket.socketpair()
    monkeypatch.setattr(rdma.socket, "create_connection", lambda *args: client)
    monkeypatch.setattr(rdma, "_configure_socket", lambda wire: None)
    monkeypatch.setattr(rdma, "checkpoint_metadata", lambda model: {"id": "checkpoint"})
    monkeypatch.setattr(rdma, "_physical_device_id", lambda logical: {0: 7}[logical])
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    captured = []

    def serve():
        captured.append(rdma._receive(server))
        rdma._send(server, {"error": "stop after attach"})
        server.close()

    worker = threading.Thread(target=serve)
    worker.start()
    with pytest.raises(RuntimeError, match="stop after attach"):
        rdma.RemoteReader("model", "owner:123", capacity=1)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert captured[0]["physical_device"] == 7
    assert "device" not in captured[0]


def test_same_nic_reader_uses_another_owner_device(rdma, monkeypatch):
    owner = object.__new__(rdma.SharedOwner)
    owner.server_address = ("10.0.0.1", 29500)
    owner.device = 0
    owner.physical_device = 10
    owner.engine = object()
    owner.engine_endpoint = "10.0.0.1:29501"
    owner.library_path = None
    owner.fabric = "roce"
    owner.listen_port = 26666
    owner.local_comm_res = None
    owner.alternate_engine = owner.alternate_endpoint = None
    owner.lock = threading.Lock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device_count=lambda: 2), raising=False)
    created = []

    def engine(endpoint, device, library_path, **kwargs):
        created.append((endpoint, device, kwargs))
        return object()

    monkeypatch.setattr(rdma, "HixlEngine", engine)
    assert owner.engine_for_reader("10.0.0.2", 10) == (owner.engine, owner.engine_endpoint)
    # Different visibility masks can map logical 0 to different physical IDs.
    assert owner.engine_for_reader("10.0.0.1", 11) == (owner.engine, owner.engine_endpoint)
    alternate = owner.engine_for_reader("10.0.0.1", 10)
    assert alternate[1] == "10.0.0.1:29502"
    assert owner.engine_for_reader("10.0.0.1", 10) == alternate
    assert created == [("10.0.0.1:29502", 1, {"listen_port": 26666, "local_comm_res": None})]


@pytest.mark.parametrize("abrupt", [False, True])
def test_owner_retains_unacknowledged_stage(rdma, monkeypatch, abrupt):
    """A control EOF does not establish completion of a remote HIXL read."""
    owner = object.__new__(rdma.SharedOwner)
    owner.metadata = {"tables": {"1": {"row_bytes": 264}}}
    owner.generation = "test-generation"
    owner.sessions, owner.retained, owner.lock, owner.pending_handlers = {}, [], threading.Lock(), 0
    stage, handle, deregistered = SimpleNamespace(data_ptr=lambda: 123), object(), []
    engine = SimpleNamespace(
        device=0,
        register=lambda tensor: handle,
        deregister=lambda value: deregistered.append(value),
    )
    owner.engine_for_reader = lambda address, device: (engine, "owner:29501")
    monkeypatch.setattr(rdma, "_configure_socket", lambda wire: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda device: None), raising=False)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: stage)
    client, server = socket.socketpair()
    worker = threading.Thread(target=rdma._OwnerConnection, args=(server, ("127.0.0.1", 1), owner))
    worker.start()
    try:
        rdma._send(client, {"op": "attach", "metadata": owner.metadata, "capacity": 1, "physical_device": 4})
        ready = rdma._receive(client)
        assert ready["generation"] == owner.generation
        if not abrupt:
            rdma._send(client, {"op": "close", "generation": owner.generation})
            assert rdma._receive(client) == {"closed": True}
    finally:
        client.close()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not owner.sessions
    if abrupt:
        assert len(owner.retained) == 1
        assert owner.retained[0] == (engine, stage, handle)
        assert not deregistered
        with pytest.raises(RuntimeError, match="retained Engram leases"):
            owner.server_close()
    else:
        assert owner.retained == []
        assert deregistered == [handle]


@pytest.mark.parametrize("physical_device", [None, -1, True, "0"])
def test_owner_rejects_missing_or_invalid_physical_device(rdma, monkeypatch, physical_device):
    owner = object.__new__(rdma.SharedOwner)
    owner.metadata = {"tables": {"1": {"row_bytes": 264}}}
    owner.sessions, owner.retained, owner.lock = {}, [], threading.Lock()
    monkeypatch.setattr(rdma, "_configure_socket", lambda wire: None)
    client, server = socket.socketpair()
    worker = threading.Thread(target=rdma._OwnerConnection, args=(server, ("127.0.0.1", 1), owner))
    worker.start()
    try:
        rdma._send(
            client, {"op": "attach", "metadata": owner.metadata, "capacity": 1, "physical_device": physical_device}
        )
        assert "physical device ID" in rdma._receive(client)["error"]
    finally:
        client.close()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not owner.sessions and not owner.retained


def test_shutdown_tracks_attach_until_abrupt_disconnect(rdma, monkeypatch):
    owner = object.__new__(rdma.SharedOwner)
    owner.metadata = {"tables": {"1": {"row_bytes": 264}}}
    owner.generation = "test-generation"
    owner.sessions, owner.retained, owner.lock, owner.pending_handlers = {}, [], threading.Lock(), 0
    started, release = threading.Event(), threading.Event()
    closed = []
    stage = SimpleNamespace(data_ptr=lambda: 123)

    def register(tensor):
        started.set()
        assert release.wait(5)
        return object()

    engine = SimpleNamespace(device=0, register=register, close=lambda: closed.append(True))
    owner.engine = engine
    owner.alternate_engine = None
    owner.engine_for_reader = lambda address, device: (engine, "owner:29501")
    monkeypatch.setattr(rdma, "_configure_socket", lambda wire: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda device: None), raising=False)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: stage)
    rdma.socketserver.ThreadingTCPServer.__init__(owner, ("127.0.0.1", 0), rdma._OwnerConnection)
    service = threading.Thread(target=owner.serve_forever)
    service.start()
    client = socket.create_connection(owner.server_address, timeout=5)
    try:
        rdma._send(client, {"op": "attach", "metadata": owner.metadata, "capacity": 1, "physical_device": 4})
        assert started.wait(5)
        owner.shutdown()
        service.join(timeout=5)
        assert not service.is_alive()
        assert owner.pending_handlers == 1 and not owner.sessions
        with pytest.raises(RuntimeError, match="retained Engram leases"):
            owner.server_close()
        assert not closed
        release.set()
        assert rdma._receive(client)["generation"] == owner.generation
        client.close()
        owner._threads.join()
        assert owner.pending_handlers == 0 and len(owner.retained) == 1
        with pytest.raises(RuntimeError, match="retained Engram leases"):
            owner.server_close()
        assert not closed
    finally:
        release.set()
        client.close()
        owner.retained.clear()
        owner.server_close()


def test_capacity_error_keeps_reader_closable(rdma, monkeypatch):
    reader = object.__new__(rdma.RemoteReader)
    reader.closed = reader.failed = False
    reader._operation_lock = threading.Lock()
    reader.capacity = 1
    reader.metadata = {"tables": {"1": {"rows": 4, "row_bytes": 264}}}
    reader.generation = "generation"
    calls = []
    reader._last_use = SimpleNamespace(synchronize=lambda: calls.append("sync"))
    reader.engine = SimpleNamespace(close=lambda: calls.append("engine.close"))
    reader.wire = SimpleNamespace(close=lambda: calls.append("wire.close"))
    reader.buffer = object()
    monkeypatch.setattr(rdma, "_send", lambda wire, msg: calls.append(msg))
    monkeypatch.setattr(rdma, "_receive", lambda wire: {"closed": True})
    table = SimpleNamespace(
        _checkpoint_key="layers.1.weight",
        num_embeddings=4,
        head_start=0,
        part_n_hash_cols=1,
        vocab_start_idx=0,
        vocab_end_idx=4,
    )
    indices = torch.tensor([[0], [1]])
    with pytest.raises(rdma.EngramCapacityError):
        reader.lookup(table, indices, None, cpu_ids=np.array([[0], [1]]))
    assert not calls and not reader.failed
    reader.close()
    assert reader.closed and calls == [
        "sync",
        "engine.close",
        {"op": "close", "generation": "generation"},
        "wire.close",
    ]


@pytest.mark.parametrize(
    "reader_path",
    [
        ("_engram_remote_reader",),
        ("model", "_engram_remote_reader"),
        ("language_model", "model", "_engram_remote_reader"),
    ],
)
@pytest.mark.parametrize("close_fails", [False, True])
def test_model_init_failure_retires_attached_reader(rdma, monkeypatch, reader_path, close_fails):
    path = Path(rdma.__file__).resolve().parents[1] / "engram_lifecycle.py"
    spec = importlib.util.spec_from_file_location("engram_lifecycle_test", path)
    assert spec is not None and spec.loader is not None
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    calls = []
    reader = object.__new__(rdma.RemoteReader)
    reader.closed = reader.failed = False
    reader._operation_lock = threading.Lock()
    reader._last_use = None
    reader.engine = SimpleNamespace(close=lambda: calls.append("engine.close"))
    reader.wire = SimpleNamespace(close=lambda: calls.append("wire.close"))
    reader.buffer = object()
    reader.generation = "generation"

    def send(wire, message):
        calls.append("lease.close")
        if close_fails:
            raise OSError("connection lost")

    monkeypatch.setattr(rdma, "_send", send)
    monkeypatch.setattr(rdma, "_receive", lambda wire: {"closed": True})

    class Model:
        @lifecycle.close_engram_on_init_failure(reader_path)
        def __init__(self):
            current: object = self
            for name in reader_path[:-1]:
                child = SimpleNamespace()
                setattr(current, name, child)
                current = child
            setattr(current, reader_path[-1], reader)
            raise ValueError("original model error")

    try:
        with pytest.raises(ValueError, match="original model error"):
            Model()
        assert calls[:2] == ["engine.close", "lease.close"]
        assert reader.failed is close_fails
        assert reader.closed is not close_fails
        if close_fails:
            assert reader in rdma._held_readers
        else:
            assert calls[-1] == "wire.close"
    finally:
        rdma._held_readers.discard(reader)


def test_model_init_cleanup_ignores_unattached_and_success(rdma):
    path = Path(rdma.__file__).resolve().parents[1] / "engram_lifecycle.py"
    spec = importlib.util.spec_from_file_location("engram_lifecycle_test", path)
    assert spec is not None and spec.loader is not None
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    calls = []

    class Model:
        @lifecycle.close_engram_on_init_failure(("_engram_remote_reader",))
        def __init__(self, fail):
            if fail:
                raise ValueError("before attach")
            self._engram_remote_reader = SimpleNamespace(close=lambda: calls.append("closed"))

    with pytest.raises(ValueError, match="before attach"):
        Model(True)
    Model(False)
    assert not calls


def test_capacity_for_batch_keeps_default_and_expands_per_layer(rdma):
    assert rdma.capacity_for_batch(65536, 8192, [32, 32], 4) == 65536
    assert rdma.capacity_for_batch(65536, 9000, [32, 32], 4) == 72000
    assert rdma.capacity_for_batch(100000, 8192, [32, 32], 4) == 100000
    assert rdma.capacity_for_batch(65536, 32768, [32], 4) == 262144
    with pytest.raises(ValueError, match="262152.*262144"):
        rdma.capacity_for_batch(65536, 32769, [32], 4)
    with pytest.raises(ValueError, match="divisible"):
        rdma.capacity_for_batch(65536, 8192, [31], 4)


def test_lookup_plan_accepts_larger_capacity_and_rejects_overflow(rdma):
    tables = {"1": np.empty((1, 264), dtype=np.uint8)}
    plan, count = rdma._lookup_plan({"op": "lookup", "layer": "1", "count": 72000}, tables, 72000)
    assert count == 72000 and plan[0][1] == 72000
    with pytest.raises(ValueError, match="lookup shape"):
        rdma._lookup_plan({"op": "lookup", "layer": "1", "count": 72001}, tables, 72000)


def test_owner_ready_file_failure_stops_service(rdma, monkeypatch, tmp_path):
    stop = threading.Event()
    calls = []
    constructed = []

    class Owner:
        generation = "test"
        metadata = {}
        tables = {}
        sessions = {}
        retained = []
        pending_handlers = 0
        lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            constructed.append(True)

        def serve_forever(self):
            stop.wait()
            calls.append("thread exited")

        def shutdown(self):
            calls.append("shutdown")
            stop.set()

        def server_close(self):
            calls.append("server_close")

    for name in (
        "vllm_ascend",
        "vllm_ascend.models",
        "vllm_ascend.models.deepseek_v41",
        "vllm_ascend.models.deepseek_v41.engram",
        "vllm_ascend.ops",
        "torch_npu",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["vllm_ascend"], "ops", sys.modules["vllm_ascend.ops"], raising=False)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.models.deepseek_v41.engram.shared_rdma", SimpleNamespace(SharedOwner=Owner)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda device: None), raising=False)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "owner",
            "--backend",
            "a2",
            "--model",
            "model",
            "--endpoint",
            "a:1",
            "--engine-endpoint",
            "a:2",
            "--ready-file",
            str(tmp_path / "ready.json"),
        ],
    )
    cli_path = Path(rdma.__file__).resolve().parents[3] / "shared_engram_rdma.py"
    spec = importlib.util.spec_from_file_location("test_engram_owner_cli", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def fail_write(path, *args, **kwargs):
        raise PermissionError("ready file cannot be written")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(PermissionError, match="ready file cannot be written"):
        cli.main()
    assert calls == ["shutdown", "thread exited", "server_close"]
    sys.argv[-1] = str(tmp_path / "missing" / "ready.json")
    with pytest.raises(FileNotFoundError, match="parent directory"):
        cli.main()
    assert constructed == [True]


@pytest.mark.parametrize("close_fails", [False, True])
def test_owner_table_load_failure_closes_or_retains_engine(rdma, monkeypatch, tmp_path, close_fails):
    (tmp_path / "index.json").write_text(json.dumps({"weight_map": {}}))
    monkeypatch.setattr(
        rdma,
        "checkpoint_metadata",
        lambda model: {"index_file": "index.json", "tables": {"1": {"rows": 1, "row_bytes": 264}}},
    )
    monkeypatch.setattr(rdma, "_physical_device_id", lambda device: device)
    calls = []

    class Engine:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            calls.append("close")
            if close_fails:
                raise OSError("engine cleanup failed")

    monkeypatch.setattr(rdma, "HixlEngine", Engine)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: (_ for _ in ()).throw(MemoryError("table failed")))
    try:
        with pytest.raises(MemoryError, match="table failed"):
            rdma.SharedOwner(str(tmp_path), "127.0.0.1:0", "127.0.0.1:1")
        assert calls == ["close"]
        assert len(rdma._held_owners) == int(close_fails)
    finally:
        rdma._held_owners.clear()


def test_owner_bind_failure_closes_socket_and_engine(rdma, monkeypatch, tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"weight_map": {}}))
    monkeypatch.setattr(rdma, "checkpoint_metadata", lambda model: {"index_file": "index.json", "tables": {}})
    monkeypatch.setattr(rdma, "_physical_device_id", lambda device: device)
    closed, sockets = [], []
    monkeypatch.setattr(rdma, "HixlEngine", lambda *args, **kwargs: SimpleNamespace(close=lambda: closed.append(True)))

    def fail_bind(owner):
        sockets.append(owner.socket)
        raise OSError("bind failed")

    monkeypatch.setattr(rdma.socketserver.TCPServer, "server_bind", fail_bind)
    with pytest.raises(OSError, match="bind failed"):
        rdma.SharedOwner(str(tmp_path), "127.0.0.1:0", "127.0.0.1:1")
    assert closed == [True] and sockets[0].fileno() == -1


def test_ub_reader_reuses_owner_engine_without_second_npu(rdma, monkeypatch):
    owner = object.__new__(rdma.SharedOwner)
    owner.server_address = ("10.0.0.1", 29500)
    owner.device = 0
    owner.engine = object()
    owner.engine_endpoint = "10.0.0.1:29501"
    owner.fabric = "ub"
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device_count=lambda: 1), raising=False)
    assert owner.engine_for_reader("10.0.0.1", 0) == (owner.engine, owner.engine_endpoint)


def test_both_layers_keep_token_and_head_order_in_one_copy(rdma):
    hashes = torch.tensor(
        [[[0, 10, -1, 13], [20, 21, 22, 23]], [[30, 31, 32, 33], [40, -1, 42, 43]]], dtype=torch.int32
    )
    copied = rdma.RemoteReader.copy_hashes(hashes, 1, 2)
    assert copied.dtype == np.int64
    np.testing.assert_array_equal(copied[:, 0], [[10, -1], [31, 32]])
    np.testing.assert_array_equal(copied[:, 1], [[21, 22], [-1, 42]])


def test_batch_reader_falls_back_without_touching_transport(rdma):
    reader = object.__new__(rdma.RemoteReader)
    reader.closed = reader.failed = False
    reader._operation_lock = threading.Lock()
    reader.supports_batch = False
    tables = [SimpleNamespace(part_n_hash_cols=6)] * 2
    hashes = torch.empty((4, 2, 24), dtype=torch.int64)
    assert reader.lookup_layers(tables, hashes, None) is None
    reader.supports_batch = True
    reader.capacity = 47
    assert reader.lookup_layers(tables, hashes, None) is None


def test_close_fences_only_owned_buffer_before_retiring_lease(rdma, monkeypatch):
    order = []
    reader = object.__new__(rdma.RemoteReader)
    reader.closed = False
    reader.failed = False
    reader._operation_lock = threading.Lock()
    reader._last_use = SimpleNamespace(synchronize=lambda: order.append("last_use"))
    reader.engine = SimpleNamespace(close=lambda: order.append("engine"))
    reader.wire = SimpleNamespace(close=lambda: order.append("wire"))
    reader.generation = "generation"
    reader.buffer = object()

    def no_device_fence():
        raise AssertionError("unrelated device work must not gate lease retirement")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=no_device_fence), raising=False)
    monkeypatch.setattr(rdma, "_send", lambda *args: order.append("owner_close"))
    monkeypatch.setattr(rdma, "_receive", lambda *args: {"closed": True})
    reader.close()
    reader.close()
    assert order == ["last_use", "engine", "owner_close", "wire"]
    assert reader.closed and reader.buffer is None


def test_failed_read_retains_buffer_and_rejects_reuse(rdma, monkeypatch):
    import weakref

    reader = object.__new__(rdma.RemoteReader)
    reader.closed = reader.failed = False
    reader._operation_lock = threading.Lock()
    reader.buffer = object()
    reader._lookup_ids = lambda *args: ("1", {"row_bytes": 264}, np.array([1], dtype=np.int64))
    reader._transfer_rows = lambda *args: (_ for _ in ()).throw(TimeoutError("READ may be in flight"))
    with pytest.raises(TimeoutError):
        reader.lookup(None, None, None)
    assert reader.failed and reader in rdma._held_readers and reader.buffer is not None
    with pytest.raises(RuntimeError, match="failed or closed"):
        reader.lookup(None, None, None)
    with pytest.raises(RuntimeError, match="external recovery"):
        reader.close()
    reference = weakref.ref(reader)
    del reader
    assert reference() is not None
    rdma._held_readers.clear()


@pytest.mark.parametrize(
    "lookups",
    [
        [],
        [{"layer": "missing", "count": 1}],
        [{"layer": "1", "count": True}],
        [{"layer": "1", "count": 2}, {"layer": "1", "count": 1}],
        [{"layer": "1", "count": 4}, {"layer": "14", "count": 3}],
    ],
)
def test_batch_plan_rejects_invalid_layout_before_ids(rdma, lookups):
    tables = {"1": torch.empty((4, 264), dtype=torch.uint8), "14": torch.empty((5, 288), dtype=torch.uint8)}
    with pytest.raises(ValueError):
        rdma._lookup_plan(dict(op="lookup_batch", lookups=lookups), tables, 6)


def test_batch_native_packing_keeps_layer_offsets_and_rejects_partial_writes(rdma):
    path = os.getenv("ENGRAM_TEST_LIBRARY")
    if not path:
        pytest.skip("requires the actual native row packer")
    lib = ctypes.CDLL(path)
    lib.engram_pack_rows.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] + [ctypes.c_size_t] * 2

    def check(code):
        assert code == 0

    engine = SimpleNamespace(lib=lib, check=check)
    tables = {
        "1": torch.arange(4 * 264).to(torch.uint8).reshape(4, 264),
        "14": torch.arange(5 * 288).to(torch.uint8).reshape(5, 288),
    }
    request = dict(op="lookup_batch", lookups=[dict(layer="1", count=3), dict(layer="14", count=2)])
    plan, count = rdma._lookup_plan(request, tables, 6)
    assert count == 5 and plan == [("1", 3, 0, 0), ("14", 2, 3, 792)]
    stage = torch.full((6 * 288,), 33, dtype=torch.uint8)
    ids = np.array([3, -1, 0, 4, 1], dtype=np.int64)
    rdma._pack_lookup_rows(engine, stage, tables, plan, ids)
    expected = torch.cat(
        [tables["1"][3], torch.zeros(264, dtype=torch.uint8), tables["1"][0], tables["14"][4], tables["14"][1]]
    )
    torch.testing.assert_close(stage[: expected.numel()], expected)
    assert torch.all(stage[expected.numel() :] == 33)
    before = stage.clone()
    ids[-1] = 5
    with pytest.raises(ValueError, match="row ID"):
        rdma._pack_lookup_rows(engine, stage, tables, plan, ids)
    torch.testing.assert_close(stage, before)


@pytest.mark.skipif(not Path("/dev/davinci_manager").exists(), reason="requires the installed NPU worker runtime")
def test_worker_retires_reader_before_network_shutdown(monkeypatch):
    from vllm_ascend.worker import worker as worker_module

    order = []
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(close_shared_engram=lambda: order.append("reader")),
        shutdown=lambda: order.append("runner"),
    )
    worker.profiler = None
    worker.weight_transfer_engine = None
    monkeypatch.setattr(worker_module, "ensure_kv_transfer_shutdown", lambda: order.append("kv"))
    worker.shutdown()
    assert order == ["reader", "kv", "runner"]


def test_worker_continues_cleanup_when_reader_close_fails(rdma):
    worker_file = Path(rdma.__file__).resolve().parents[3] / "worker" / "worker.py"
    module = ast.parse(worker_file.read_text())
    worker_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    shutdown = next(node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name == "shutdown")
    events = []

    def fail_close():
        events.append("reader")
        raise RuntimeError("reader close failed")

    namespace: dict[str, Any] = {
        "AscendEplbState": type("AscendEplbState", (), {}),
        "ensure_kv_transfer_shutdown": lambda: events.append("kv"),
        "ensure_ec_transfer_shutdown": lambda: events.append("ec"),
    }
    exec(compile(ast.Module(body=[shutdown], type_ignores=[]), str(worker_file), "exec"), namespace)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(close_shared_engram=fail_close), shutdown=lambda: events.append("runner")
        ),
        profiler=None,
        weight_transfer_engine=None,
    )
    with pytest.raises(RuntimeError, match="reader close failed"):
        namespace["shutdown"](worker)
    assert events == ["reader", "kv", "ec", "runner"]


@pytest.mark.skipif(not Path("/dev/davinci_manager").exists(), reason="requires NPU kernels")
@pytest.mark.parametrize("scale_dtype,row_bytes", [("F32", 288), ("F8_E8M0", 264)])
@pytest.mark.parametrize("code_dtype", ["I8", "F8_E4M3"])
def test_packed_dequant_preserves_signed_codes_and_scales(scale_dtype, row_bytes, code_dtype):
    import torch_npu  # noqa: F401

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.ops.triton.engram_lookup import dequantize_packed_engram

    torch.npu.set_device(1 if torch.npu.device_count() > 1 else 0)
    codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(3, 1)
    if code_dtype == "F8_E4M3":
        codes = codes.float().div(8).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.125, 0.25, 0.5, 1, 2, 4, 8, 16]], dtype=torch.float32).repeat(3, 1)
    raw = (
        scales.view(torch.uint8)
        if scale_dtype == "F32"
        else torch.tensor([[124, 125, 126, 127, 128, 129, 130, 131]], dtype=torch.uint8).repeat(3, 1)
    )
    packed = torch.empty((3, row_bytes), dtype=torch.uint8)
    packed[:, :256].copy_(codes.view(torch.uint8))
    packed[:, 256:].copy_(raw)
    actual = torch.empty((3, 256), dtype=torch.bfloat16, device="npu")
    dequantize_packed_engram(packed.to("npu"), actual, 3, row_bytes, scale_dtype, code_dtype)
    torch.npu.synchronize()
    expected = (codes.float().unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_checkpoint_samples_detect_same_metadata_changed_weights(rdma, tmp_path):
    config = {"text_config": {"engram_head_dim": 256, "engram_layer_ids": [1], "engram_num_embeddings": [4]}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    index = {"weight_map": {weight_key: "weights.safetensors", scale_key: "weights.safetensors"}}
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index))
    weights = torch.arange(4 * 256).reshape(4, 256).to(torch.int8)
    scales = torch.ones((4, 8), dtype=torch.float32)
    save_file({weight_key: weights, scale_key: scales}, tmp_path / "weights.safetensors")
    # Worker initialization supplies a non-CPU default device. Sampling must
    # not materialize the table there before selecting the requested rows.
    with torch.device("meta"):
        first = rdma.checkpoint_metadata(tmp_path)
    weights[0, 0] += 1
    save_file({weight_key: weights, scale_key: scales}, tmp_path / "weights.safetensors")
    second = rdma.checkpoint_metadata(tmp_path)
    assert first["protocol"] == second["protocol"] == 2
    assert first["identity"] == second["identity"]
    assert first["tables"]["1"]["samples"] != second["tables"]["1"]["samples"]
    assert second["tables"]["1"]["row_bytes"] == 288


@pytest.mark.parametrize("width", [264, 288])
def test_native_packer_mask_order_and_no_partial_write(width):
    path = os.getenv("ENGRAM_TEST_LIBRARY")
    if not path:
        pytest.skip("build the CANN native library to test its actual packer")
    lib = ctypes.CDLL(path)
    lib.engram_pack_rows.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] + [ctypes.c_size_t] * 2
    source = np.arange(4 * width, dtype=np.uint8).reshape(4, width)
    output = np.full((4, width), 33, dtype=np.uint8)
    ids = np.array([3, -1, 0, 3], dtype=np.int64)
    args = (output.ctypes.data, source.ctypes.data, width, ids.ctypes.data, len(ids), len(source))
    assert lib.engram_pack_rows(*args) == 0
    np.testing.assert_array_equal(output, np.stack([source[3], np.zeros(width, dtype=np.uint8), source[0], source[3]]))
    before = output.copy()
    ids[-1] = 4
    assert lib.engram_pack_rows(*args) != 0
    np.testing.assert_array_equal(output, before)


@pytest.mark.parametrize("index_name", ["quant_model_weights.safetensors.index.json", "model.safetensors.index.json"])
def test_fp8_checkpoint_preserves_code_dtype_and_index(rdma, tmp_path, index_name):
    config = {"text_config": {"engram_head_dim": 256, "engram_layer_ids": [1], "engram_num_embeddings": [4]}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    index = {"weight_map": {weight_key: "weights.safetensors", scale_key: "weights.safetensors"}}
    (tmp_path / index_name).write_text(json.dumps(index))
    codes = torch.arange(-128, 128).float().div(8).repeat(4, 1).to(torch.float8_e4m3fn)
    save_file({weight_key: codes, scale_key: torch.ones(4, 8)}, tmp_path / "weights.safetensors")
    metadata = rdma.checkpoint_metadata(tmp_path)
    assert metadata["index_file"] == index_name
    assert metadata["tables"]["1"]["code_dtype"] == "F8_E4M3"
    # Equal raw bytes reinterpreted as INT8 must not attach as the same table.
    save_file({weight_key: codes.view(torch.int8), scale_key: torch.ones(4, 8)}, tmp_path / "weights.safetensors")
    assert metadata != rdma.checkpoint_metadata(tmp_path)
