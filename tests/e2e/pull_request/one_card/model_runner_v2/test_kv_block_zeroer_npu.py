# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend NPU coverage for hybrid KV block zeroing."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.utils import AscendKVBlockZeroer


@pytest.mark.parametrize("ratio", [1, 3])
@pytest.mark.parametrize("padded", [False, True])
def test_zero_split_kv_blocks_preserves_other_blocks_and_padding(ratio, padded):
    device = torch.device("npu")
    init_device_properties_triton()
    num_blocks, kernel_bs = 4, 128
    spec = FullAttentionSpec(block_size=kernel_bs * ratio, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    caches = []
    expected_caches = []
    raw_buffers = []
    expected_buffers = []
    for width, dtype in ((512, torch.bfloat16), (64, torch.bfloat16), (32, torch.int8)):
        payload = kernel_bs * width
        stride = payload + (128 if padded else 0)
        raw = torch.full((128 + num_blocks * ratio * stride,), 7, dtype=dtype, device=device)
        expected = raw.clone()
        shape, strides = (num_blocks * ratio, kernel_bs, 1, width), (stride, width, width, 1)
        caches.append(torch.as_strided(raw, shape, strides, storage_offset=128))
        expected_caches.append(torch.as_strided(expected, shape, strides, storage_offset=128))
        raw_buffers.append(raw)
        expected_buffers.append(expected)
    caches.append(torch.empty(num_blocks * ratio, kernel_bs, 1, 0, dtype=torch.bfloat16, device=device))
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["attn", "shared"])
    zeroer = AscendKVBlockZeroer(device, pin_memory=True)
    zeroer.init_meta(
        [group],
        [[kernel_bs]],
        "auto",
        set(),
        {name: SimpleNamespace(kv_cache=tuple(caches)) for name in group.layer_names},
    )
    for block_ids in ([0], [1, 3]):
        zeroer.zero_block_ids(block_ids)
        for cache in expected_caches:
            for block_id in block_ids:
                cache[block_id * ratio : (block_id + 1) * ratio].zero_()
        torch.npu.synchronize()
        for actual, expected in zip(raw_buffers, expected_buffers):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_hybrid_split_views_sharing_one_allocation():
    device = torch.device("npu")
    init_device_properties_triton()
    num_blocks, block_size = 4, 128
    state_elements = 1024
    key_elements, value_elements = num_blocks * block_size * 512, num_blocks * block_size * 64
    raw = torch.full((state_elements + key_elements + value_elements,), 7, dtype=torch.bfloat16, device=device)
    expected = raw.clone()

    def views(buffer):
        k = buffer[state_elements : state_elements + key_elements].view(num_blocks, block_size, 1, 512)
        v = buffer[state_elements + key_elements :].view(num_blocks, block_size, 1, 64)
        return k, v

    spec = AscendMLAAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["attn"])
    zeroer = AscendKVBlockZeroer(device, pin_memory=True)
    zeroer.init_meta([group], [[block_size]], "auto", set(), {"attn": SimpleNamespace(kv_cache=views(raw))})
    zeroer.zero_block_ids([1, 3])
    for cache in views(expected):
        cache[1].zero_()
        cache[3].zero_()
    torch.npu.synchronize()
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


def test_abort_reallocate_zeroes_reused_blocks_before_replacement_reads():
    device = torch.device("npu")
    init_device_properties_triton()
    register_ascend_kv_cache_specs()
    # Keep exactly two allocatable blocks (plus the null block) so both
    # supported upstream allocator orders must reuse the aborted request.
    num_blocks, block_size = 3, 128
    spec = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    block_pool = BlockPool(num_blocks, enable_caching=False, hash_block_size=block_size)
    manager = get_manager_for_kv_cache_spec(
        spec,
        max_in_flight_tokens=256,
        max_model_len=4096,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
        needs_kv_cache_zeroing=True,
    )

    key_cache = torch.zeros(num_blocks, block_size, 1, 48, dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros(num_blocks, block_size, 1, 16, dtype=torch.bfloat16, device=device)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["attn"])
    zeroer = AscendKVBlockZeroer(device, pin_memory=True)
    zeroer.init_meta(
        [group],
        [[block_size]],
        "auto",
        set(),
        {"attn": SimpleNamespace(kv_cache=(key_cache, value_cache))},
    )

    original = manager.allocate_new_blocks("aborted", 129, 129)
    original_ids = manager.take_new_block_ids()
    assert original_ids == [block.block_id for block in original]
    zeroer.zero_block_ids(original_ids)
    for block_id in original_ids:
        key_cache[block_id].fill_(7)
        value_cache[block_id].fill_(7)

    manager.free("aborted")
    replacement = manager.allocate_new_blocks("replacement", 129, 129)
    reused_ids = manager.take_new_block_ids()
    assert {block.block_id for block in replacement} == set(original_ids)
    assert set(reused_ids) == set(original_ids)

    zeroer.zero_block_ids(reused_ids)
    torch.npu.synchronize()
    for block_id in reused_ids:
        torch.testing.assert_close(key_cache[block_id], torch.zeros_like(key_cache[block_id]), rtol=0, atol=0)
        torch.testing.assert_close(value_cache[block_id], torch.zeros_like(value_cache[block_id]), rtol=0, atol=0)
