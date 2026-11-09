# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for row semantics, pool isolation, and allocation ownership."""

import ctypes
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def backend(monkeypatch):
    path = Path(__file__).parents[3] / "vllm_ascend/distributed/ple_gva.py"
    spec = importlib.util.spec_from_file_location("ple_gva_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
    return module


class FakePool:
    def __init__(self):
        self.allocations = {}
        self.config = None
        self.closed = False
        self.free_error = 0

    def initialize(self, config):
        self.config = config
        return 0

    def malloc(self, size):
        buf = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(buf)
        self.allocations[ptr] = buf
        return ptr

    def free(self, ptr):
        if self.free_error:
            return self.free_error
        del self.allocations[ptr]
        return 0

    def uninitialize(self):
        self.closed = True

    def sparse_copy(self, src, dst, lengths, count, device):
        assert count.item() == src.numel()
        assert src.numel() % 2 == 0
        for s, d, length in zip(src.tolist(), dst.tolist(), lengths.tolist()):
            if length:
                ctypes.memmove(d, s, length)
        return 0


@pytest.fixture
def native(monkeypatch):
    pools = []

    def create():
        pool = FakePool()
        pools.append(pool)
        return pool

    api = SimpleNamespace(Offload=create, OffloadConfig=SimpleNamespace, Scene=SimpleNamespace(LOCAL=0, SHARED=1))
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", SimpleNamespace(offload=api))
    return api, pools


@pytest.mark.parametrize("count", [0, 1, 3, 64, 65, 129])
def test_lookup_masks_and_odd_counts(backend, native, count):
    table = backend.PLEGvaTable((4, 16), torch.device("cpu"))
    table.tensor.copy_(torch.arange(64).reshape(4, 16))
    ids = torch.tensor(([9, 10, 13, 14, -1, 2**40] * ((count + 5) // 6))[:count], dtype=torch.int64)
    actual = table.lookup(ids, 10, 14)
    expected = torch.zeros((count, 16), dtype=torch.bfloat16)
    for i, row in enumerate(ids.tolist()):
        if 10 <= row < 14:
            expected[i] = table.tensor[row - 10]
    torch.testing.assert_close(actual, expected)
    table.close()
    table.close()
    assert native[1][0].closed
    with pytest.raises(RuntimeError, match="closed"):
        table.lookup(ids, 10, 14)


def test_independent_instances(backend, native):
    first = backend.PLEGvaTable((4, 16), torch.device("cpu"))
    second = backend.PLEGvaTable((4, 16), torch.device("cpu"))
    second.tensor.fill_(7)
    first.close()
    assert not second.pool.closed
    assert torch.all(second.lookup(torch.tensor([0]), 0, 4) == 7)
    second.close()


def test_free_failure_is_retryable(backend, native):
    table = backend.PLEGvaTable((4, 16), torch.device("cpu"))
    pointer = table.pointer
    table.pool.free_error = 42
    with pytest.raises(RuntimeError, match="free failed"):
        table.close()
    assert table.pointer == pointer
    assert not table.pool.closed
    with pytest.raises(RuntimeError, match="closed"):
        table.lookup(torch.tensor([0]), 0, 4)
    table.pool.free_error = 0
    table.close()
    assert table.pointer == 0
    assert table.pool.closed


def test_old_singleton_api_is_rejected(backend, native):
    native[0].Offload = None
    with pytest.raises(RuntimeError, match="independent"):
        backend.PLEGvaTable((4, 16), torch.device("cpu"))
    assert not native[1]


@pytest.mark.parametrize("shape", [(0, 16), (4, 0), (4, 17), (4, 32768)])
def test_invalid_layout_fails_before_initialization(backend, native, shape):
    with pytest.raises(ValueError):
        backend.PLEGvaTable(shape, torch.device("cpu"))
    assert not native[1]


def test_allocation_failure_closes_pool(backend, native, monkeypatch):
    monkeypatch.setattr(FakePool, "malloc", lambda self, size: 0)
    with pytest.raises(RuntimeError, match="allocation failed"):
        backend.PLEGvaTable((4, 16), torch.device("cpu"))
    assert native[1][0].closed


def test_shared_consumer_has_no_backing(backend, native, monkeypatch):
    group = SimpleNamespace(rank_in_group=1, world_size=2, ranks=[0, 1], cpu_group=object())

    def broadcast(payload, **kwargs):
        payload[:] = [123456, None]

    monkeypatch.setattr(backend.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(backend.dist, "barrier", Mock())
    table = backend.PLEGvaTable((4, 16), torch.device("cpu"), group=group, store_url="tcp://127.0.0.1:31001")
    assert table.pool.config.alloc_size == 0
    assert table.tensor.is_meta
    assert table.source == 123456
    assert not table.pool.allocations
    table.close()


def test_shared_requires_endpoint(backend, native):
    with pytest.raises(ValueError, match="dedicated"):
        backend.PLEGvaTable((4, 16), torch.device("cpu"), group=object())
    assert not native[1]


def test_lookup_rejects_invalid_dtype_and_range(backend, native):
    table = backend.PLEGvaTable((4, 16), torch.device("cpu"))
    try:
        with pytest.raises(ValueError, match="int32 or int64"):
            table.lookup(torch.tensor([0.5]), 0, 4)
        with pytest.raises(ValueError, match="range"):
            table.lookup(torch.tensor([0]), 0, 5)
    finally:
        table.close()
