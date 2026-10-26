# SPDX-License-Identifier: Apache-2.0
"""Allocation/publication/release invariants with a fake BM, no NPU imports."""

import ctypes
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def storage(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/shared_host.py"
    spec = importlib.util.spec_from_file_location("_memfabric_storage_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device=lambda _: nullcontext(), synchronize=lambda: None), raising=False
    )
    return module


@pytest.fixture
def runtime():
    data = ctypes.create_string_buffer(65536)
    calls: list[object] = []

    class Handle:
        fail_leave = False

        def join(self):
            calls.append("join")
            return 0

        def leave(self):
            calls.append("leave")
            return -1 if self.fail_leave else 0

        def destroy(self):
            calls.append("destroy")

        def peer_rank_ptr(self, rank, kind):
            calls.append(("rank", rank))
            return 0x10000000

        def gva_to_va(self, gva, kind):
            base = ctypes.addressof(data) if kind == "host" else 0x20000000
            return base + gva - 0x10000000

        def local_mem_size(self, kind):
            return 0

    def create(*args, **kwargs):
        calls.append(("create", args, kwargs))
        return Handle()

    def check(result, name):
        if result:
            raise RuntimeError(name)

    bm = SimpleNamespace(
        create2=create,
        BmDataOpType=SimpleNamespace(SDMA=1),
        BmMemType=SimpleNamespace(HOST="global_host", LOCAL_HOST="host", LOCAL_DEVICE="device"),
    )
    return SimpleNamespace(
        bm=bm, device=torch.device("cpu"), rank=0, objects=set(), check=check, data=data, calls=calls
    )


@pytest.mark.parametrize("owner", [True, False])
def test_only_owner_contributes_rounded_dram(storage, runtime, owner):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=owner, owner_rank=3)
    call = runtime.calls[0]
    assert call[1] == (0, (1 << 30) if owner else 0, 1 << 30)
    assert call[2]["flags"] == storage.DRAM_MAP_HOST_VA
    assert ("rank", 0 if owner else 3) in runtime.calls
    buffer.close()
    assert not runtime.objects


def test_publication_uses_owner_cpu_mapping_and_preserves_bounds(storage, runtime):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=True)
    source = torch.arange(256, dtype=torch.int16).to(torch.int8).reshape(1, 256)
    buffer.publish(7, source)
    actual = torch.frombuffer(runtime.data, dtype=torch.int8)[7 * 256 : 8 * 256]
    torch.testing.assert_close(actual, source.flatten())
    with pytest.raises(ValueError, match="outside"):
        buffer.publish(32, source)
    buffer.close()


def test_live_views_block_release_and_failed_leave_is_retryable(storage, runtime):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=False, owner_rank=0)
    view = buffer.view(8, 24)
    assert view.tensor.device.type == "meta"
    assert view.ptrs.tolist() == [0x20000000 + 8 * 256]
    with pytest.raises(RuntimeError, match="slices"):
        buffer.close()
    view.close()
    buffer.handle.fail_leave = True
    with pytest.raises(RuntimeError, match="leave"):
        buffer.close()
    assert not buffer.closed and runtime.objects == {0}
    assert "destroy" not in runtime.calls
    buffer.handle.fail_leave = False
    buffer.close()
    assert buffer.closed and runtime.calls[-1] == "destroy"


def test_consumer_cannot_publish_or_duplicate_live_object(storage, runtime):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=False, owner_rank=0)
    with pytest.raises(RuntimeError, match="owner"):
        buffer.publish(0, torch.zeros(1, 256, dtype=torch.int8))
    with pytest.raises(ValueError, match="Duplicate"):
        storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=False, owner_rank=0)
    buffer.close()


def test_publication_resolves_each_independent_slice(storage, runtime, monkeypatch):
    monkeypatch.setattr(storage, "GIB", 64)
    buffer = storage.MappedHostBuffer(runtime, 0, 4, 64, torch.int8, owner=True)
    chunks = [ctypes.create_string_buffer(64) for _ in range(4)]

    def resolve(gva, kind):
        offset = gva - buffer.gva
        return ctypes.addressof(chunks[offset // 64]) + offset % 64

    buffer.handle.gva_to_va = resolve
    source = torch.arange(256).to(torch.int8).view(4, 64)
    buffer.publish(0, source)
    actual = torch.cat([torch.frombuffer(chunk, dtype=torch.int8) for chunk in chunks])
    torch.testing.assert_close(actual, source.flatten())
    buffer.close()


def test_noncontiguous_npu_chunk_rejected_before_pointer_publication(storage, runtime):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=False, owner_rank=0)

    def resolve(gva, kind):
        offset = gva - buffer.gva
        return 0x20000000 + offset + ((1 << 30) if offset >= 4096 else 0)

    buffer.handle.gva_to_va = resolve
    with pytest.raises(RuntimeError, match="not contiguous"):
        buffer.view(8, 24)
    assert not buffer._views
    buffer.close()
