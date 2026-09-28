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
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
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
def test_cache_adaptor_uses_sk_and_preserves_backing_storage(dtype, index_dtype, tokens, layout):
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
        patch.object(torch_npu, "npu_scatter_pa_cache") as pa,
        patch.object(torch_npu, "npu_scatter_nd_update_") as generic,
    ):
        assert DeviceOperator.scatter_cache(key, cache, slots, tokens) is None
        torch.npu.synchronize()
        sk.assert_called_once()
        pa.assert_not_called()
        generic.assert_not_called()

    assert cache.data_ptr() == pointer
    torch.testing.assert_close(backing.view(torch.uint8).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "key_dtype,scale_dtype", [(torch.bfloat16, None), (torch.int8, torch.float16), (torch.float8_e4m3fn, torch.float32)]
)
@pytest.mark.parametrize("row_gap", [False, True])
@pytest.mark.parametrize("tokens", [3, 2049])
@torch.inference_mode()
def test_indexer_cache_uses_sk_for_keys_and_scales(key_dtype, scale_dtype, row_gap, tokens):
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
        _use_c8_reshape_optim=lambda: False,
    )
    with (
        patch.object(
            torch.ops._C_ascend, "npu_scatter_nd_update_sk", wraps=torch.ops._C_ascend.npu_scatter_nd_update_sk
        ) as sk,
        patch.object(torch_npu, "npu_scatter_nd_update_") as generic,
        patch.object(torch_npu, "npu_scatter_pa_cache") as pa,
    ):
        AscendSFAIndexerBackend.write_cache(
            indexer,
            keys[0] if scale_dtype else keys[0].unsqueeze(1),
            keys[1] if scale_dtype else None,
            slots,
            SimpleNamespace(num_actual_tokens=tokens // 2),
        )
        torch.npu.synchronize()
        assert sk.call_count == len(caches)
        generic.assert_not_called()
        pa.assert_not_called()
    for backing, reference in zip(backings, expected):
        torch.testing.assert_close(backing.view(torch.uint8).cpu(), reference, rtol=0, atol=0)
