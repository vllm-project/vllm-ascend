# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.pcp_kv_cache import _copy_pcp_kv_cache_kernel, copy_pcp_kv_cache


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "case,strided",
    [("empty", False), ("padding", False), ("mixed", True), ("many", False), ("large", True)],
)
def test_c8_cache_preserves_packed_bytes(dtype, case, strided):
    # Include every byte pattern, including FP8 NaNs: the RoPE and scale
    # sections are raw BF16/FP32 bytes, not FP8 values.
    shape = (20, 32, 1, 656)
    raw = (torch.arange(20 * 32 * 656, dtype=torch.int32) % 256 - 128).to(torch.int8).reshape(shape)
    source = raw.to("npu")
    if strided:
        raw = raw[::2]
        source = source[::2]
    source = source.view(dtype)
    indices = {
        "empty": [],
        "padding": [-1, -1, -1],
        "mixed": [0, 31, -1, 32, 319, -1],
        "many": list(range(257)) + [-1],
        "large": list(range(319)) * 2 + [-1],
    }[case]
    # Non-contiguous slot mappings are accepted too.
    slots = torch.tensor([[i, -1] for i in indices], dtype=torch.int64, device="npu").reshape(-1, 2)[:, 0]
    packed = copy_pcp_kv_cache((source,), slots)
    expected = torch.zeros((len(indices), 656), dtype=torch.int8)
    for row, slot in enumerate(indices):
        if slot >= 0:
            expected[row] = raw[slot // 32, slot % 32, 0]
    assert packed.dtype == torch.int8
    assert torch.equal(packed.cpu(), expected)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("num_slots", [5, 128, 257])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_contiguous_slot_slices_reuse_kernel(c8, num_slots, slot_dtype):
    dtype = torch.int8 if c8 else torch.bfloat16
    dims = (656,) if c8 else (512, 64)
    cache = tuple(
        (torch.arange(20 * 32 * dim, dtype=torch.int32) % 127).to(dtype).reshape(20, 32, 1, dim).to("npu")
        for dim in dims
    )
    gold = torch.cat([tensor.cpu().reshape(-1, dim) for tensor, dim in zip(cache, dims)], dim=-1)
    indices = torch.arange(num_slots + 4, dtype=slot_dtype) % (20 * 32)
    indices[1::5] = -1
    slot_storage = indices.to("npu")
    device = torch.npu.current_device()
    # Isolate the in-memory cache so an existing unaligned variant cannot hide
    # an extra specialization. Compiled disk artifacts can still be reused.
    if hasattr(_copy_pcp_kv_cache_kernel, "device_caches"):
        kernel_cache = _copy_pcp_kv_cache_kernel.device_caches[device][0]
    else:
        kernel_cache = _copy_pcp_kv_cache_kernel.cache[device]
    with patch.dict(kernel_cache, clear=True):
        for offset in range(5):
            slots = slot_storage[offset : offset + num_slots]
            assert slots.is_contiguous()
            assert slots.data_ptr() == slot_storage.data_ptr() + offset * slots.element_size()
            actual = copy_pcp_kv_cache(cache, slots).cpu()
            selected = indices[offset : offset + num_slots].long()
            expected = gold[selected.clamp_min(0)].clone()
            expected[selected < 0] = 0
            assert torch.equal(actual, expected)
            assert len(kernel_cache) == 1


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_slots", [5, 257, 513])
@pytest.mark.parametrize("strided", [False, True])
def test_separate_latent_and_rope_cache(dtype, num_slots, strided):
    k = torch.arange(4 * 8 * 512, dtype=torch.float32).reshape(4, 8, 1, 512)
    r = torch.arange(4 * 8 * 64, dtype=torch.float32).reshape(4, 8, 1, 64)
    k, r = k.to(dtype).to("npu"), r.to(dtype).to("npu")
    if strided:
        k, r = k[::2], r[::2]
    indices = (torch.arange(num_slots, dtype=torch.int64) * 7) % (k.shape[0] * k.shape[1])
    indices[1::5] = -1
    slots = indices.to("npu")
    packed = copy_pcp_kv_cache((k, r), slots)
    expected = torch.cat((k.cpu().reshape(-1, 512), r.cpu().reshape(-1, 64)), dim=-1)[indices.clamp_min(0)]
    expected[indices < 0] = 0
    assert torch.equal(packed.cpu(), expected)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("layout", ["component_major", "token_fused"])
def test_current_mla_kv_read_does_not_copy_history_cache(layout):
    # A full NoPE copy is 256 MiB, while the current 8193 rows need about
    # 18 MiB including packed output and both dense components. Use a wide
    # margin for allocator rounding while still rejecting a full cache copy.
    blocks, block_size, key_dim, rope_dim, tokens = 2048, 128, 512, 64, 8193
    page_elements = block_size * (key_dim + rope_dim)
    storage = torch.empty(blocks * page_elements, dtype=torch.bfloat16, device="npu")
    if layout == "component_major":
        key_stride, rope_stride = key_dim, rope_dim
        rope_offset = block_size * key_dim
    else:
        key_stride = rope_stride = key_dim + rope_dim
        rope_offset = key_dim
    key = torch.as_strided(storage, (blocks, block_size, 1, key_dim), (page_elements, key_stride, key_stride, 1))
    rope = torch.as_strided(
        storage, (blocks, block_size, 1, rope_dim), (page_elements, rope_stride, rope_stride, 1), rope_offset
    )
    key.fill_(3)
    rope.fill_(7)
    indices = (torch.arange(tokens, dtype=torch.int64) * 127) % (blocks * block_size)
    indices[::5] = -1
    slots = indices.to("npu")
    # Compile and initialize the kernel before measuring live tensor memory.
    warmup = AscendMLAImpl._gather_current_kv_from_cache((key, rope), slots)
    torch.npu.synchronize()
    del warmup
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    current_key, current_rope = AscendMLAImpl._gather_current_kv_from_cache((key, rope), slots)
    torch.npu.synchronize()
    peak_bytes = torch.npu.max_memory_allocated() - baseline
    assert peak_bytes < key.numel() * key.element_size() // 2, f"Current KV read allocated {peak_bytes} bytes"
    for actual, dim, value in ((current_key, key_dim, 3), (current_rope, rope_dim, 7)):
        expected = torch.full((tokens, 1, dim), value, dtype=torch.bfloat16)
        expected[indices < 0] = 0
        assert actual.is_contiguous()
        assert torch.equal(actual.cpu(), expected)
    # The read must not alter either view of the shared storage.
    assert torch.all(key == 3).item()
    assert torch.all(rope == 7).item()
