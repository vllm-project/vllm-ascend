# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("tokens", [3, 2049])
@torch.inference_mode()
def test_native_main_c8_cache_preserves_packing_and_padding(tokens):
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    padded = ((tokens + 7) // 8) * 8
    packed = (torch.arange(padded * 656, device="npu").reshape(padded, 656) % 251 - 125).to(torch.int8)
    k_nope, k_pe, scale = packed.split([512, 128, 16], dim=-1)
    cache = torch.full((32, 128, 1, 656), -7, dtype=torch.int8, device="npu")
    slots = torch.arange(padded, dtype=torch.int32, device="npu") + 37
    slots[tokens:] = -1
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.sfa_qsfa_packed_kv_head_dim = 656
    impl._store_parallel_kv(
        k_pe, k_nope, scale, None, [], (cache,), slots, SimpleNamespace(num_actual_tokens=tokens), False
    )
    torch.npu.synchronize()
    reference = torch.full_like(cache, -7, device="cpu")
    reference.view(-1, 656)[slots[:tokens].cpu().long()] = packed[:tokens].cpu()
    torch.testing.assert_close(cache.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,width", [(torch.int8, 656), (torch.int8, 128), (torch.bfloat16, 128), (torch.float16, 1)]
)
@pytest.mark.parametrize("tokens", [2049, 8193])
@pytest.mark.parametrize("layout", ["contiguous", "row_gap", "offset"])
@torch.inference_mode()
def test_platform_cache_store_preserves_all_backing_bytes(dtype, width, tokens, layout):
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    blocks, block_size = 128, 128
    backing = torch.full((blocks + 1, block_size, 1, width * 2), -7, dtype=dtype, device="npu")
    if layout == "row_gap":
        cache = backing[:-1, ..., :width]
    elif layout == "offset":
        cache = backing.view(2 * (blocks + 1), block_size, 1, width)[1 : blocks + 1]
    else:
        cache = backing.view(2 * (blocks + 1), block_size, 1, width)[:blocks]
    initial = backing.cpu()
    reference = torch.as_strided(initial, cache.shape, cache.stride(), cache.storage_offset())
    padded = ((tokens + 7) // 8) * 8
    key = (torch.arange(padded * width * 2, device="npu").reshape(padded, width * 2) % 251 - 125).to(dtype)[:, ::2]
    slots = torch.empty(padded * 2, device="npu", dtype=torch.int32)[::2]
    logical = torch.arange(tokens, device="npu", dtype=torch.int32) + 37
    slots[:tokens] = (blocks - 2 - logical // block_size) * block_size + logical % block_size
    slots[tokens:] = -1
    ptr = cache.data_ptr()
    for delta in (0, 1):
        key.add_(delta)
        reference.view(-1, width)[slots[:tokens].cpu().long()] = key[:tokens].cpu()
        DeviceOperator.try_scatter_cache(key, cache, slots, tokens)
        torch.npu.synchronize()
        assert cache.data_ptr() == ptr
        torch.testing.assert_close(backing.cpu(), initial, rtol=0, atol=0)


@torch.inference_mode()
def test_unsupported_fast_dtype_preserves_fallback_with_negative_slots():
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    key = torch.ones(2048, 128, dtype=torch.float32, device="npu")
    cache = torch.zeros(32, 128, 1, 128, dtype=torch.float32, device="npu")
    slots = torch.arange(2048, dtype=torch.int32, device="npu")
    meta = SimpleNamespace(num_actual_tokens=2048)
    slots[-3:] = -1
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.is_kv_producer, impl.is_kv_consumer = True, False
    impl._store_parallel_kv(None, None, None, key, [], (cache,), slots, meta, False)
    torch.npu.synchronize()
    reference = torch.zeros_like(cache, device="cpu").view(-1, 128)
    reference[:2045] = 1
    torch.testing.assert_close(cache.cpu().view(-1, 128), reference, rtol=0, atol=0)


@torch.inference_mode()
def test_unsupported_inner_stride_matches_generic_scatter_behavior():
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    key = torch.ones(8, 128, dtype=torch.int8, device="npu")
    backing = torch.full((2, 128, 1, 256), -7, dtype=torch.int8, device="npu")
    reference = backing.clone()
    cache, reference_cache = backing[..., ::2], reference[..., ::2]
    slots = torch.arange(8, dtype=torch.int32, device="npu")
    try:
        torch_npu.npu_scatter_nd_update_(reference_cache.view(-1, 128), slots.view(-1, 1), key)
        torch.npu.synchronize()
    except RuntimeError:
        with pytest.raises(RuntimeError):
            DeviceOperator.try_scatter_cache(key, cache, slots, 8)
            torch.npu.synchronize()
    else:
        DeviceOperator.try_scatter_cache(key, cache, slots, 8)
        torch.npu.synchronize()
        torch.testing.assert_close(backing.cpu(), reference.cpu(), rtol=0, atol=0)
