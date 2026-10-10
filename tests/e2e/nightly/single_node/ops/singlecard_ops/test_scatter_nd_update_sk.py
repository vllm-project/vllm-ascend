import gc
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.device.device_config import check_ascend_device_type
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import bootstrap_custom_op_env

# A5 的 hardware profile 未启用 RUNTIME_CUSTOM_OPS，enable_custom_op() 会直接
# 返回 False，需参考 test_add_rms_norm_bias_a5.py 用 bootstrap + 显式 import。
bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, E402

check_ascend_device_type()

seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def scatter_nd_update_golden(var, indices, update):
    """CPU reference of npu_scatter_nd_update_sk.

    var/indices/update are CPU tensors. var is [a, b], indices is [n, 1],
    update is [n, b]. Indices are unique (or updates for duplicated indices
    are identical), so the row-by-row assignment is deterministic.
    """
    out = var.clone()
    for i in range(indices.shape[0]):
        out[int(indices[i, 0].item())] = update[i]
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("width,count", [(1, 128), (128, 128), (656, 2049)])
@pytest.mark.parametrize("row_gap", [False, True])
def test_a5_cache_destination_layout(dtype, width, count, row_gap):
    if not get_current_hardware_profile().supports(HardwareCapability.SCATTER_ND_FP8_CACHE_STORE):
        pytest.skip("Requires A5 FP8 cache kernels")
    # Keep indices and updates contiguous: only the destination can have gaps.
    # Rows beyond `width` reproduce the arch35 non-contiguous storage-bound bug.
    backing = torch.full((8192, width * (2 if row_gap else 1)), 7.0, device="npu").to(dtype)
    cache = backing[:, :width]
    indices = (torch.arange(count, device="npu", dtype=torch.int64) + 37).view(-1, 1)
    updates = torch.arange(count * width, device="npu").remainder(17).reshape(count, width).float().to(dtype)
    expected = backing.cpu()
    expected[37 : 37 + count, :width] = updates.cpu()
    assert cache.is_contiguous() == (not row_gap)
    assert indices.is_contiguous() and updates.is_contiguous()

    with (
        patch.object(
            torch.ops._C_ascend, "npu_scatter_nd_update_sk", wraps=torch.ops._C_ascend.npu_scatter_nd_update_sk
        ) as sk,
        patch.object(torch_npu, "npu_scatter_nd_update_", wraps=torch_npu.npu_scatter_nd_update_) as generic,
        patch.object(torch_npu, "npu_scatter_pa_cache", wraps=torch_npu.npu_scatter_pa_cache) as pa,
    ):
        DeviceOperator.scatter_cache(cache, indices, updates)
    sk.assert_not_called()
    assert pa.call_count == int(not row_gap)
    assert generic.call_count == int(row_gap)
    if row_gap:
        assert all(actual is original for actual, original in zip(generic.call_args.args, (cache, indices, updates)))
    else:
        assert pa.call_args.kwargs["key_cache"].data_ptr() == cache.data_ptr()
    # Check the entire allocation, including row gaps and untouched cache rows.
    assert torch.equal(backing.view(torch.uint8).cpu(), expected.view(torch.uint8))


@pytest.mark.parametrize(
    "a",
    [16, 77],
)
@pytest.mark.parametrize(
    "b",
    [8, 128],
)
@pytest.mark.parametrize(
    "var_dtype, idx_dtype",
    [
        (torch.float16, torch.int32),
        (torch.bfloat16, torch.int64),
        (torch.float32, torch.int32),
        (torch.int8, torch.int64),
    ],
)
@pytest.mark.parametrize(
    "contiguous",
    [True, False],
)
def test_scatter_nd_update_sk(a: int, b: int, var_dtype, idx_dtype, contiguous: bool):
    n = max(1, a // 4)

    # unique indices in [0, a)
    idx = np.random.choice(a, size=n, replace=False)
    indices_cpu = torch.from_numpy(idx.astype(np.int64)).view(-1, 1).to(idx_dtype)

    if var_dtype == torch.int8:
        update_cpu = torch.randint(-32, 32, (n, b), dtype=torch.int32).to(torch.int8)
    else:
        update_cpu = torch.randn(n, b, dtype=torch.float32).to(var_dtype)

    # var is a non-contiguous view (rows of width b in a buffer of width 2*b)
    # when contiguous=False, matching the real KV-cache layout.
    row_stride = b if contiguous else 2 * b
    var_cpu = torch.zeros(a, row_stride, dtype=var_dtype)[:, :b]

    var_npu = var_cpu.clone().npu()
    torch.ops._C_ascend.npu_scatter_nd_update_sk(var_npu, indices_cpu.npu(), update_cpu.npu())

    golden = scatter_nd_update_golden(var_cpu, indices_cpu, update_cpu)
    # pure data movement, bit-exact comparison
    assert torch.equal(var_npu.cpu(), golden)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "var_dtype, idx_dtype",
    [
        (torch.float16, torch.int32),
        (torch.bfloat16, torch.int64),
        (torch.int8, torch.int64),
    ],
)
@pytest.mark.parametrize(
    "contiguous",
    [True, False],
)
def test_scatter_nd_update_sk_duplicate_indices(var_dtype, idx_dtype, contiguous: bool):
    """Duplicated indices take the sort path; the result is deterministic when
    all updates for the same index are identical."""
    a, b, n = 32, 64, 8
    row = 7  # duplicated target row
    indices_cpu = torch.full((n, 1), row, dtype=idx_dtype)

    if var_dtype == torch.int8:
        update_cpu = torch.randint(-32, 32, (1, b), dtype=torch.int32).to(torch.int8).repeat(n, 1)
    else:
        update_cpu = torch.randn(1, b, dtype=torch.float32).to(var_dtype).repeat(n, 1)

    row_stride = b if contiguous else 2 * b
    var_cpu = torch.zeros(a, row_stride, dtype=var_dtype)[:, :b]

    var_npu = var_cpu.clone().npu()
    torch.ops._C_ascend.npu_scatter_nd_update_sk(var_npu, indices_cpu.npu(), update_cpu.npu())

    golden = scatter_nd_update_golden(var_cpu, indices_cpu, update_cpu)
    assert torch.equal(var_npu.cpu(), golden)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens", [3, 2049])
@pytest.mark.parametrize("layout", ["contiguous", "row_gap", "offset"])
@torch.inference_mode()
def test_cache_adaptor_dispatch_preserves_backing_storage(dtype, index_dtype, tokens, layout):
    # Run this same adaptor regression on A2/A3 and A5, including the packed
    # SFA C8 width and genuinely strided NPU views (do not clone the views).
    width, block_size, blocks = 656, 128, 32
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if fp8 and not get_current_hardware_profile().supports(HardwareCapability.SCATTER_ND_FP8_CACHE_STORE):
        pytest.skip("FP8 SK cache writes require arch35")
    storage_dtype = torch.uint8 if fp8 else dtype
    backing = torch.full(
        (blocks + 1, block_size, 1, width * 2), 0xA5 if fp8 else -7, dtype=storage_dtype, device="npu"
    ).view(dtype)
    if layout == "row_gap":
        cache = backing[:-1, ..., :width]
    elif layout == "offset":
        cache = backing.view(-1, block_size, 1, width)[1 : blocks + 1]
    else:
        cache = backing.view(-1, block_size, 1, width)[:blocks]
    padded = ((tokens + 7) // 8) * 8
    key = torch.arange(padded * width * 2, device="npu").reshape(padded, width * 2) % 251 - 125
    key = key.to(storage_dtype).view(dtype)[:, ::2]
    slots = torch.empty(padded * 2, dtype=index_dtype, device="npu")[::2]
    slots[:tokens] = torch.arange(tokens, dtype=index_dtype, device="npu") + 37
    slots[tokens:] = -1
    expected = backing.view(torch.uint8).cpu()
    expected_cache = torch.as_strided(expected.view(dtype), cache.shape, cache.stride(), cache.storage_offset())
    expected_cache.view(-1, width).view(torch.uint8)[slots[:tokens].cpu().long()] = (
        key[:tokens].contiguous().view(torch.uint8).cpu()
    )
    pointer = cache.data_ptr()

    with (
        patch.object(
            torch.ops._C_ascend, "npu_scatter_nd_update_sk", wraps=torch.ops._C_ascend.npu_scatter_nd_update_sk
        ) as sk,
        patch.object(
            torch_npu, "npu_scatter_pa_cache", wraps=getattr(torch_npu, "npu_scatter_pa_cache", None), create=True
        ) as pa,
        patch.object(torch_npu, "npu_scatter_nd_update_", wraps=torch_npu.npu_scatter_nd_update_) as generic,
    ):
        assert DeviceOperator.scatter_cache(cache.view(-1, key.shape[-1]), slots.view(-1, 1), key) is None
        torch.npu.synchronize()
        a5 = get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED
        expected_sk = not a5
        expected_pa = a5 and layout != "row_gap"
        assert sk.call_count == int(expected_sk)
        assert pa.call_count == int(expected_pa)
        assert generic.call_count == int(not (expected_sk or expected_pa))

    assert cache.data_ptr() == pointer
    torch.testing.assert_close(backing.view(torch.uint8).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "key_dtype,scale_dtype", [(torch.bfloat16, None), (torch.int8, torch.float16), (torch.float8_e4m3fn, torch.float32)]
)
@pytest.mark.parametrize("row_gap", [False, True])
@pytest.mark.parametrize("tokens", [3, 2049])
@torch.inference_mode()
def test_indexer_cache_dispatch_for_keys_and_scales(key_dtype, scale_dtype, row_gap, tokens):
    if key_dtype == torch.float8_e4m3fn and not get_current_hardware_profile().supports(
        HardwareCapability.SCATTER_ND_FP8_CACHE_STORE
    ):
        pytest.skip("FP8 indexer cache requires arch35")
    padded = ((tokens + 7) // 8) * 8
    slots = torch.arange(padded, dtype=torch.int64, device="npu") + 37
    slots[tokens // 2] = -1  # Invalid slots inside a gathered region must be skipped.
    slots[tokens:] = -1
    cpu_slots = slots.cpu().long()
    valid = cpu_slots >= 0
    keys, caches, backings, expected = [], [], [], []
    for dtype, width in [(key_dtype, 128)] + ([(scale_dtype, 1)] if scale_dtype else []):
        storage_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
        backing = torch.full((32, 128, 1, width * (2 if row_gap else 1)), 7, dtype=storage_dtype, device="npu").view(
            dtype
        )
        cache = backing[..., :width]
        key = (
            torch.arange(padded * width, device="npu")
            .reshape(padded, width)
            .remainder(17)
            .to(storage_dtype)
            .view(dtype)
        )
        before = backing.view(torch.uint8).cpu()
        reference = torch.as_strided(before.view(dtype), cache.shape, cache.stride(), cache.storage_offset())
        reference.view(-1, width).view(torch.uint8)[cpu_slots[valid]] = key.view(torch.uint8).cpu()[valid]
        keys.append(key)
        caches.append(cache)
        backings.append(backing)
        expected.append(before)
    indexer = SimpleNamespace(
        enable_sparse_li_c8=scale_dtype is not None,
        k_cache=SimpleNamespace(kv_cache=tuple(caches)),
    )
    with (
        patch.object(
            torch.ops._C_ascend, "npu_scatter_nd_update_sk", wraps=torch.ops._C_ascend.npu_scatter_nd_update_sk
        ) as sk,
        patch.object(torch_npu, "npu_scatter_nd_update_", wraps=torch_npu.npu_scatter_nd_update_) as generic,
        patch.object(
            torch_npu, "npu_scatter_pa_cache", wraps=getattr(torch_npu, "npu_scatter_pa_cache", None), create=True
        ) as pa,
    ):
        AscendSFAIndexerBackend.write_cache(
            indexer,
            keys[0] if scale_dtype else keys[0].unsqueeze(1),
            keys[1] if scale_dtype else None,
            slots,
        )
        torch.npu.synchronize()
        a5 = get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED
        expected_sk = not a5
        expected_pa = a5 and not row_gap
        assert sk.call_count == (len(caches) if expected_sk else 0)
        assert generic.call_count == (0 if expected_sk or expected_pa else len(caches))
        assert pa.call_count == (len(caches) if expected_pa else 0)
    for backing, reference in zip(backings, expected):
        torch.testing.assert_close(backing.view(torch.uint8).cpu(), reference, rtol=0, atol=0)
