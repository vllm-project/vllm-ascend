"""Runtime contract for an NPU view of registered pinned CPU memory."""

import gc
import importlib
import os
import weakref
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch

pytest.importorskip("torch_npu")
importlib.import_module("vllm_ascend.vllm_ascend_C")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

try:
    triton_ascend_version = version("triton-ascend")
except PackageNotFoundError:
    pytest.skip("requires Triton-Ascend", allow_module_level=True)

pytestmark = [
    pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"),
    pytest.mark.skipif(
        "pinned_mem_register:True" not in os.getenv("PYTORCH_NPU_ALLOC_CONF", ""),
        reason="requires mapped pinned CPU memory",
    ),
]
requires_triton_uva = pytest.mark.skipif(
    triton_ascend_version in ("3.2.1", "3.2.2"),
    reason="installed Triton-Ascend launcher rejects mapped host memory",
)


@triton.jit
def _read_view(src, dst, stride: tl.constexpr, n: tl.constexpr, block: tl.constexpr):
    indices = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(src + indices * stride, mask=indices < n, other=0)
    tl.store(dst + indices, values, mask=indices < n)


def _read_on_npu(view: torch.Tensor) -> torch.Tensor:
    result = torch.empty(view.numel(), dtype=view.dtype, device="npu")
    _read_view[(triton.cdiv(view.numel(), 32),)](view, result, view.stride(0), view.numel(), 32)
    return result.cpu()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_npu_view_preserves_metadata(dtype: torch.dtype):
    base = torch.arange(64, dtype=dtype).pin_memory()
    cpu = base[3:58:2]
    assert torch.ops._C_ascend.can_get_npu_view_from_cpu_tensor(cpu)
    view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu)

    assert view.device.type == "npu"
    assert view.shape == cpu.shape
    assert view.stride() == cpu.stride()
    assert view.dtype == cpu.dtype


def test_npu_views_keep_cpu_storage_alive():
    def make_views():
        base = torch.arange(32, dtype=torch.int32).pin_memory()
        cpu = base[1::2]
        return (
            weakref.ref(cpu),
            torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu),
            torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu),
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


@requires_triton_uva
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_npu_view_reads_cpu_updates_without_copy(dtype: torch.dtype):
    base = torch.arange(64, dtype=dtype).pin_memory()
    cpu = base[3:58:2]
    view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu)
    torch.testing.assert_close(_read_on_npu(view), cpu)

    cpu.add_(100)
    torch.testing.assert_close(_read_on_npu(view), cpu)


@requires_triton_uva
def test_npu_view_keeps_cpu_storage_alive():
    def make_view():
        base = torch.arange(32, dtype=torch.int32).pin_memory()
        cpu = base[1::2]
        return weakref.ref(cpu), torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu)

    cpu_ref, view = make_view()
    gc.collect()
    assert cpu_ref() is not None
    torch.testing.assert_close(_read_on_npu(view), torch.arange(1, 32, 2, dtype=torch.int32))

    del view
    gc.collect()
    assert cpu_ref() is None


def test_empty_npu_view():
    cpu = torch.empty((0, 4), dtype=torch.int32, pin_memory=True)
    view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu)
    assert view.device.type == "npu"
    assert view.shape == cpu.shape
    assert view.stride() == cpu.stride()
    assert view.dtype == cpu.dtype


def test_npu_view_rejects_unpinned_or_device_input():
    assert not torch.ops._C_ascend.can_get_npu_view_from_cpu_tensor(torch.ones(4))
    with pytest.raises(RuntimeError, match="pinned CPU memory"):
        torch.ops._C_ascend.get_npu_view_from_cpu_tensor(torch.ones(4))
    with pytest.raises(RuntimeError, match="CPU tensor"):
        torch.ops._C_ascend.get_npu_view_from_cpu_tensor(torch.ones(4, device="npu"))
