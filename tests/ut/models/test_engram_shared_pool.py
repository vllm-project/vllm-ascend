# SPDX-License-Identifier: Apache-2.0
"""Real TCP lifecycle tests with a fake allocation backend; no NPU needed."""

import importlib
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
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.fail_release = False

        def view(self, start, end):
            return types.SimpleNamespace(close=lambda: None)

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
