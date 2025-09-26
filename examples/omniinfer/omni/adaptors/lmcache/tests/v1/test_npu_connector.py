# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

# Standard
import random

# Third Party
import pytest
import torch

from tests.utils import (
    generate_kv_cache_paged_list_tensors_tuple,
    check_paged_kv_cache_equal_with_tuple
)

# First Party
import ascend_lmcache.adapter.adapter

from lmcache.v1.memory_management import (
    MemoryFormat,
    PinMemoryAllocator,
)

from ascend_lmcache.v1.npu_connector import VLLMPagedMemNPUConnectorV2

def get_block_num(slot_mapping, start, end, block_size):
    return (slot_mapping[start:end] // block_size).unique().numel()

@pytest.mark.parametrize("slop_type", ["random", "Sequence"])
def test_vllm_paged_connector_v3_deepseek(slop_type):
    num_blocks = 25
    block_size = 128
    num_layers = 61
    num_heads = 1
    head_size = 576
    lora = 512
    rope = 64
    device = "npu"
    hidden_dim = num_heads * head_size

    num_tokens = 2049
    chunk_size = 256

    allocator = PinMemoryAllocator(num_layers * hidden_dim * chunk_size * block_size * 2)

    gpu_kv_src = generate_kv_cache_paged_list_tensors_tuple(
        num_blocks, device, block_size, num_layers, lora, rope
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors_tuple(
        num_blocks, device, block_size, num_layers, lora, rope
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    if slop_type == "Sequence":
        start = random.randint(0, num_blocks * block_size - num_tokens - 1)
        slot_mapping = range(start, start + num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

    # Check the npu kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal_with_tuple(
            gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, lora, rope, block_size, num_blocks
        )

    connector = VLLMPagedMemNPUConnectorV2(
        hidden_dim,
        num_layers,
        block_size=block_size,
        use_gpu=False,
        chunk_size=chunk_size,
        dtype=gpu_kv_src[0][0].dtype,
        device=device,
        use_mla=True,
    )
    connector2 = VLLMPagedMemNPUConnectorV2(
        hidden_dim,
        num_layers,
        block_size=block_size,
        use_gpu=False,
        chunk_size=chunk_size,
        dtype=gpu_kv_src[0][0].dtype,
        device=device,
        use_mla=True,
    )
    assert connector.use_mla == True
    assert connector2.use_mla == True

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(get_block_num(slot_mapping, start, end, block_size))
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)

        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
            offset=0,
        )
        assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    check_paged_kv_cache_equal_with_tuple(
        gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, lora, rope, block_size, num_blocks
    )