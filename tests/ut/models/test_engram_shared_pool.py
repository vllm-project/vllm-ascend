# SPDX-License-Identifier: Apache-2.0
"""Real TCP lifecycle tests with a fake allocation backend; no NPU needed."""

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def pool_module(monkeypatch):
    package = types.ModuleType("_shared_pool_ut")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = importlib.import_module("_shared_pool_ut.shared_pool")

    class Buffer:
        def __init__(self, runtime=None, object_id=None, rows=0, width=0, dtype=None, **kwargs):
            self.closed = False
            self.fail_release = False
            self.rows, self.width, self.dtype = rows, width, dtype

        def view(self, start, end):
            return types.SimpleNamespace(
                tensor=torch.empty((end - start, self.width), dtype=self.dtype, device="meta"),
                close=lambda: None,
            )

        def close(self):
            if self.fail_release:
                raise RuntimeError("device release failed")
            self.closed = True

    monkeypatch.setattr(module, "MappedHostBuffer", Buffer)

    class Runtime:
        @staticmethod
        def acquire(*args, **kwargs):
            return types.SimpleNamespace(rank=0, close=lambda: None)

    monkeypatch.setattr(module, "MemfabricRuntime", Runtime)
    yield module
    for name in list(sys.modules):
        if name.startswith("_shared_pool_ut."):
            sys.modules.pop(name)


@pytest.fixture
def owner(pool_module, tmp_path):
    owner = pool_module.SharedPoolOwner(tmp_path / "pool", "model-a", device="npu:0")
    owner.add("codes", 32, 256, torch.int8)
    owner.ready()
    yield owner
    owner.close(timeout=2)


def test_identity_rejection_does_not_hold_lease(pool_module, owner):
    with pytest.raises(ValueError, match="model-compatible"):
        pool_module.SharedPoolClient(owner.root, "model-b", device="npu:0")
    owner.close(timeout=1)
    assert owner.closed


def test_bfloat16_tensor_round_trips_through_owner_manifest_and_client_view(pool_module, tmp_path):
    owner = pool_module.SharedPoolOwner(tmp_path / "bf16-pool", "model-bf16", device="npu:0")
    owner.add("embedding", 32, 8, torch.bfloat16)
    owner.ready()
    client = pool_module.SharedPoolClient(owner.root, "model-bf16", device="npu:0")
    try:
        assert client.manifest["tensors"]["embedding"]["dtype"] == "bfloat16"
        view = client.view("embedding", 4, 12, rows=32, width=8, dtype=torch.bfloat16)
        assert view.tensor.dtype == torch.bfloat16
        assert view.tensor.shape == (8, 8)
        with pytest.raises(ValueError, match="layout mismatch"):
            client.view("embedding", 4, 12, rows=32, width=8, dtype=torch.float32)
    finally:
        client.close()
        owner.close(timeout=1)


def test_drain_waits_for_consumer_and_rejects_new_attach(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(TimeoutError):
            owner.close()
        assert not owner.buffers["codes"].closed
        with pytest.raises(ValueError, match="not ready"):
            pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    finally:
        client.close()
    owner.close(timeout=1)
    assert owner.buffers["codes"].closed


def test_manifest_failure_keeps_live_lease_and_rejects_new_attach(pool_module, owner, monkeypatch):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    original = owner._publish_manifest

    def fail_publish():
        raise OSError("disk full")

    monkeypatch.setattr(owner, "_publish_manifest", fail_publish)
    try:
        with pytest.raises(TimeoutError):
            owner.close(timeout=0.05)
        assert owner.manifest["state"] == "draining"
        assert not owner.buffers["codes"].closed
        with pytest.raises(ValueError, match="not ready"):
            pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
        client.close()
        owner.close(timeout=1)
        assert owner.closed and owner.buffers["codes"].closed
    finally:
        monkeypatch.setattr(owner, "_publish_manifest", original)
        client.close()


def test_failed_device_release_retains_lease_until_retry(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
    buffer = client.buffers["codes"]
    try:
        buffer.fail_release = True
        with pytest.raises(RuntimeError, match="device release failed"):
            client.close()
        with pytest.raises(TimeoutError):
            owner.close()
        assert not owner.buffers["codes"].closed
    finally:
        buffer.fail_release = False
        client.close()
    owner.close(timeout=1)


def test_client_failure_releases_other_buffers_and_retries(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
    failing = client.buffers["codes"]
    other = pool_module.MappedHostBuffer()
    client.buffers["other"] = other
    failing.fail_release = True
    with pytest.raises(RuntimeError, match="device release failed"):
        client.close()
    assert other.closed and not failing.closed and not client.closed
    with pytest.raises(TimeoutError):
        owner.close()
    failing.fail_release = False
    client.close()
    assert client.closed and failing.closed
    owner.close(timeout=1)


def test_failed_mapping_construction_keeps_client_lease_until_cleanup(pool_module, owner, monkeypatch):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    pending = pool_module.MappedHostBuffer()
    pending.fail_release = True

    def fail_mapping(*args, **kwargs):
        raise pool_module.MappingCleanupError(pending)

    monkeypatch.setattr(pool_module, "MappedHostBuffer", fail_mapping)
    with pytest.raises(pool_module.MappingCleanupError):
        client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
    with pytest.raises(RuntimeError, match="device release failed"):
        client.close()
    with pytest.raises(TimeoutError):
        owner.close()
    pending.fail_release = False
    client.close()
    owner.close(timeout=1)
    assert client.closed and owner.closed


def test_failed_owner_mapping_construction_is_retained_for_retry(pool_module, tmp_path, monkeypatch):
    owner = pool_module.SharedPoolOwner(tmp_path / "pool", "model-a", device="npu:0")
    pending = pool_module.MappedHostBuffer()
    pending.fail_release = True

    def fail_mapping(*args, **kwargs):
        raise pool_module.MappingCleanupError(pending)

    monkeypatch.setattr(pool_module, "MappedHostBuffer", fail_mapping)
    with pytest.raises(pool_module.MappingCleanupError):
        owner.add("codes", 32, 256, torch.int8)
    with pytest.raises(RuntimeError, match="device release failed"):
        owner.close()
    assert not owner.closed
    pending.fail_release = False
    owner.close()
    assert owner.closed and pending.closed


def test_owner_failure_releases_other_buffers_and_retries(pool_module, owner):
    failing = owner.buffers["codes"]
    other = pool_module.MappedHostBuffer()
    owner.buffers["other"] = other
    failing.fail_release = True
    with pytest.raises(RuntimeError, match="device release failed"):
        owner.close()
    assert other.closed and not failing.closed and not owner.closed
    failing.fail_release = False
    owner.close(timeout=1)
    assert owner.closed and failing.closed


def test_shutdown_retry_after_model_deletion_releases_tcp_lease(pool_module, owner):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    shutdown = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    embedding = types.SimpleNamespace(_shared_pool_client=client, close_host_offload=client.close)

    class Parent:
        def shutdown(self):
            if hasattr(self, "model"):
                del self.model
                raise RuntimeError("empty cache failed")

    runner_class = ast.parse("class Runner(Parent):\n    pass\n").body[0]
    assert isinstance(runner_class, ast.ClassDef)
    runner_class.body = [shutdown]
    tree = ast.fix_missing_locations(ast.Module(body=[runner_class], type_ignores=[]))
    ns = dict(Parent=Parent, C8KVBlockZeroer=type("C8KVBlockZeroer", (), {}))
    exec(compile(tree, str(path), "exec"), ns)
    runner = ns["Runner"]()
    runner.model = types.SimpleNamespace(modules=lambda: [embedding])
    with pytest.raises(RuntimeError, match="empty cache failed"):
        runner.shutdown()
    assert not client.closed
    runner.shutdown()
    assert client.closed
    owner.close(timeout=1)
    assert owner.closed


def test_layout_rejected_before_mapping(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(ValueError, match="layout mismatch"):
            client.view("codes", 0, 16, rows=33, width=256, dtype=torch.int8)
        assert not client.buffers
    finally:
        client.close()


def test_consumer_reattaches_without_reloading_owner(pool_module, owner):
    original = owner.buffers["codes"]
    for _ in range(2):
        client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
        client.close()
    assert owner.buffers["codes"] is original
    assert not original.closed


def test_private_directory_required(pool_module, owner):
    owner.root.chmod(0o755)
    with pytest.raises(ValueError, match="mode0700"):
        pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    owner.root.chmod(0o700)


def test_old_schema_rejected_and_lease_released(pool_module, owner):
    owner.manifest["schema"] = 1
    with pytest.raises(ValueError, match="manifest"):
        pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    owner.close(timeout=1)
    assert owner.closed


def test_invalid_bm_object_rejected_before_mapping(pool_module, owner):
    owner.manifest["tensors"]["codes"]["object_id"] = -1
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(ValueError, match="object id"):
            client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
        assert not client.buffers
        assert client.runtime is None
    finally:
        client.close()
