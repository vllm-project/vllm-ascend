# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Standard
import random

# Third Party
import pytest
import torch
from tests.utils import generate_kv_cache_paged_list_tensors, check_paged_kv_cache_equal

# First Party
from lmcache.v1.memory_management import PinMemoryAllocator
import ascend_lmcache.c_ops as lmc_ops

def test_from_npu_to_cpu():
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "npu"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    npu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size, num_layers, num_heads, head_size)
    npu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size, num_layers, num_heads, head_size)
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
    allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(npu_kv_src, npu_kv_dst, num_tokens, slot_mapping, num_heads, head_size)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = torch.Size([2, num_layers, num_tokens, hidden_dim])
        memory_obj = allocator.allocate(shape, npu_kv_src[0][0].dtype)

        kv_cache_pointers_src = torch.empty(num_layers, dtype=torch.int64, device="cpu")
        kv_cache_pointers_src.numpy()[:] = [t.data_ptr() for t in npu_kv_src]
        page_buffer_size = npu_kv_src[0].shape[1] * npu_kv_src[0].shape[2]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers_src,
            slot_mapping[start:end],
            npu_kv_src[0].device,
            page_buffer_size,
            True,
            False,
        )

        kv_cache_pointers_dst = torch.empty(num_layers, dtype=torch.int64, device="cpu")
        kv_cache_pointers_dst.numpy()[:] = [t.data_ptr() for t in npu_kv_dst]
        page_buffer_size = npu_kv_dst[0].shape[1] * npu_kv_dst[0].shape[2]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers_dst,
            slot_mapping[start:end],
            npu_kv_dst[0].device,
            page_buffer_size,
            False,
            False,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()
    check_paged_kv_cache_equal(
        npu_kv_src, npu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
    )