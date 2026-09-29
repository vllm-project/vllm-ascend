# SPDX-License-Identifier: Apache-2.0
"""UVA view and MRV2 wrapper component tests."""

import gc
import importlib
import os
import weakref
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu import buffer_utils

pytest.importorskip("torch_npu")
importlib.import_module("vllm_ascend.vllm_ascend_C")
patch_uva = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_uva")

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

try:
    triton_ascend_version = version("triton-ascend")
except PackageNotFoundError:
    triton_ascend_version = None

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"
)
requires_registered_pinned = pytest.mark.skipif(
    "pinned_mem_register:True" not in os.getenv("PYTORCH_NPU_ALLOC_CONF", ""),
    reason="requires mapped pinned CPU memory",
)
requires_triton_uva = pytest.mark.skipif(
    triton is None or triton_ascend_version is None or triton_ascend_version in ("3.2.1", "3.2.2"),
    reason="requires a Triton-Ascend launcher that accepts mapped host memory",
)


if triton is not None:

    @triton.jit
    def _read_view(src, dst, stride: tl.constexpr, n: tl.constexpr, block: tl.constexpr):
        indices = tl.program_id(0) * block + tl.arange(0, block)
        values = tl.load(src + indices * stride, mask=indices < n, other=0)
        tl.store(dst + indices, values, mask=indices < n)


def _mapped_view(cpu: torch.Tensor) -> torch.Tensor:
    view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu)
    assert view is not None
    return view


def _read_on_npu(view: torch.Tensor) -> torch.Tensor:
    result = torch.empty(view.numel(), dtype=view.dtype, device="npu")
    _read_view[(triton.cdiv(view.numel(), 32),)](view, result, view.stride(0), view.numel(), 32)
    return result.cpu()


@requires_registered_pinned
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_npu_view_preserves_metadata(dtype: torch.dtype):
    base = torch.arange(64, dtype=dtype).pin_memory()
    cpu = base[3:58:2]
    view = _mapped_view(cpu)

    assert view.device.type == "npu"
    assert view.shape == cpu.shape
    assert view.stride() == cpu.stride()
    assert view.dtype == cpu.dtype


@requires_registered_pinned
def test_npu_views_keep_cpu_storage_alive():
    def make_views():
        base = torch.arange(32, dtype=torch.int32).pin_memory()
        cpu = base[1::2]
        return (
            weakref.ref(cpu),
            _mapped_view(cpu),
            _mapped_view(cpu),
        )

    cpu_ref, first, second = make_views()
    gc.collect()
    assert cpu_ref() is not None

    del first
    gc.collect()
    assert cpu_ref() is not None

    del second
    gc.collect()
    assert cpu_ref() is None


@requires_registered_pinned
@requires_triton_uva
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_npu_view_reads_cpu_updates_without_copy(dtype: torch.dtype):
    base = torch.arange(64, dtype=dtype).pin_memory()
    cpu = base[3:58:2]
    view = _mapped_view(cpu)
    torch.testing.assert_close(_read_on_npu(view), cpu)

    cpu.add_(100)
    torch.testing.assert_close(_read_on_npu(view), cpu)


@requires_registered_pinned
@requires_triton_uva
def test_npu_view_keeps_cpu_storage_alive():
    def make_view():
        base = torch.arange(32, dtype=torch.int32).pin_memory()
        cpu = base[1::2]
        return weakref.ref(cpu), _mapped_view(cpu)

    cpu_ref, view = make_view()
    gc.collect()
    assert cpu_ref() is not None
    torch.testing.assert_close(_read_on_npu(view), torch.arange(1, 32, 2, dtype=torch.int32))

    del view
    gc.collect()
    assert cpu_ref() is None


@requires_registered_pinned
def test_empty_npu_view():
    cpu = torch.empty((0, 4), dtype=torch.int32, pin_memory=True)
    view = _mapped_view(cpu)
    assert view.device.type == "npu"
    assert view.shape == cpu.shape
    assert view.stride() == cpu.stride()
    assert view.dtype == cpu.dtype


def test_npu_view_returns_none_for_unpinned_input():
    assert torch.ops._C_ascend.get_npu_view_from_cpu_tensor(torch.ones(4)) is None
    with pytest.raises(RuntimeError, match="CPU tensor"):
        torch.ops._C_ascend.get_npu_view_from_cpu_tensor(torch.ones(4, device="npu"))


def test_fallback_copies_modified_prefix_and_sparse_rows(monkeypatch):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    buffer = patch_uva.UvaBufferWrapper((4, 2), torch.int32)

    buffer.cpu[:2] = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    torch.testing.assert_close(buffer.uva().cpu(), buffer.cpu._tensor)

    buffer.cpu[3] = torch.tensor([5, 6], dtype=torch.int32)
    torch.testing.assert_close(buffer.uva().cpu(), buffer.cpu._tensor)


def test_unmapped_storage_uses_fallback(monkeypatch):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    monkeypatch.setattr(torch.ops._C_ascend, "get_npu_view_from_cpu_tensor", lambda _: None)
    buffer = patch_uva.UvaBufferWrapper((2, 2), torch.int32)

    assert not buffer._use_real_uva
    buffer.np[0] = [7, 8]
    torch.testing.assert_close(buffer.uva().cpu(), buffer.cpu._tensor)


@pytest.mark.parametrize("max_concurrency", [2, 3])
@pytest.mark.parametrize("input_type", ["list", "numpy", "tensor"])
def test_pool_fallback_growth_shrink_and_round_robin(monkeypatch, max_concurrency, input_type):
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    pool = buffer_utils.UvaBufferPool((2, 2), torch.int32, max_concurrency=max_concurrency)

    for step, length in enumerate((2, 5, 1, 6, 3)):
        expected = torch.arange(length * 2, dtype=torch.int32).reshape(length, 2) + step
        values = expected.tolist() if input_type == "list" else expected.numpy() if input_type == "numpy" else expected
        before = list(pool._uva_bufs)
        slot = (pool._curr + 1) % max_concurrency

        result = pool.copy_to_uva(values)
        assert pool._curr == slot
        assert result.shape == expected.shape
        torch.testing.assert_close(result.cpu(), expected)
        for other_slot in range(max_concurrency):
            if other_slot != slot:
                assert pool._uva_bufs[other_slot] is before[other_slot]
        if length <= before[slot].cpu.shape[0]:
            assert pool._uva_bufs[slot] is before[slot]
        else:
            assert pool._uva_bufs[slot].cpu.shape[0] == 1 << (length - 1).bit_length()


@requires_registered_pinned
def test_real_path_uses_npu_typed_view(monkeypatch):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    buffer = patch_uva.UvaBufferWrapper((4, 2), torch.int32)

    assert buffer._use_real_uva
    assert buffer.cpu.device.type == "cpu"
    assert buffer.uva().device.type == "npu"
    assert buffer.uva().shape == buffer.cpu.shape


@requires_registered_pinned
def test_pool_real_path_returns_mapped_view(monkeypatch):
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    pool = buffer_utils.UvaBufferPool((2, 2), torch.int32, max_concurrency=2)

    result = pool.copy_to_uva(np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert result.device.type == "npu"
    expected_view = _mapped_view(pool._uva_bufs[pool._curr].cpu)
    assert result.data_ptr() == expected_view.data_ptr()
