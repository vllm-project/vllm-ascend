import gc
import time

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch.set_printoptions(threshold=np.inf)

enable_custom_op()


def cal_slot(key, key_cache, slot_mapping, block_size):
    key_expect = key_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        block_index = slot // block_size
        block_offset = slot % block_size
        key_expect[block_index][block_offset] = token_key
    return key_expect.npu()


def cal_scatternd(key, key_cache, slot_mapping, block_size):
    key_expect = key_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        key_expect[slot] = token_key

    return key_expect.npu()


@pytest.mark.parametrize("num_tokens", [16])  # 6398
@pytest.mark.parametrize("num_head", [1])  # 512
@pytest.mark.parametrize("block_size", [128])  # 128
@pytest.mark.parametrize("num_blocks", [1773])  # 1599
@pytest.mark.parametrize("count", [1])
def test_siso(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k = 1
    key = torch.rand((num_tokens, num_head, head_size_k), dtype=torch.float16).npu()
    # key = torch.randint(low=0,high=128,size=(num_tokens,head_size_k), dtype=torch.int8 )
    key_cache = torch.rand((num_blocks, block_size, num_head, head_size_k), dtype=torch.float16).npu()
    # key_cache = torch.randint(low=0,high=128,size=(num_blocks, block_size, num_head,head_size_k), dtype=torch.int8 )

    slot_list = []
    for i in range(0, num_tokens):
        slot_list.append(2 + i)
    assert num_tokens == len(slot_list)
    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_slot(key, key_cache, slot_mapping_npu, block_size)
    warm_up = 0

    for _ in range(warm_up):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    N = 101

    for _ in range(N):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)

    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1)


@pytest.mark.parametrize("num_tokens", [16])  # 6398
@pytest.mark.parametrize("num_head", [1])  # 512
@pytest.mark.parametrize("block_size", [128])  # 128
@pytest.mark.parametrize("num_blocks", [1773])  # 1599
@pytest.mark.parametrize("count", [1])
def test_scatter(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k = 64
    key = torch.randint(low=0, high=128, size=(num_tokens, num_head, head_size_k), dtype=torch.int8).npu()
    # key = torch.rand((num_tokens, num_head,head_size_k), dtype=torch.float16).npu()

    key_cache = torch.randint(
        low=0, high=128, size=(num_blocks * block_size, num_head, head_size_k), dtype=torch.int8
    ).npu()
    # key_cache = torch.rand((num_blocks* block_size, num_head,head_size_k), dtype=torch.float16).npu()
    slot_list = []
    for i in range(0, num_tokens):
        slot_list.append([2 + i])

    assert num_tokens == len(slot_list)
    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_scatternd(key, key_cache, slot_mapping_npu, block_size)
    N = 101
    for i in range(N):
        torch_npu.npu_scatter_nd_update_(key_cache, slot_mapping_npu, key)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1)


@pytest.mark.parametrize("num_tokens", [16])  # 6398
@pytest.mark.parametrize("num_head", [1])  # 512
@pytest.mark.parametrize("block_size", [128])  # 128
@pytest.mark.parametrize("num_blocks", [1773])  # 1599
@pytest.mark.parametrize("count", [1])
def test_myops(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k = 64
    # key_cache = torch.rand((num_blocks, block_size, num_head,head_size_k), dtype=torch.float16)
    key_cache = torch.randint(low=0, high=128, size=(num_blocks, block_size, num_head, head_size_k), dtype=torch.int8)
    key_cache_npu = key_cache.npu()

    slot_list = []
    for i in range(0, num_tokens):
        slot_list.append(2 + i)

    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()
    # slot_mapping_cpu = slot_mapping_npu.to("cpu",non_blocking=True)
    # num_draft_tensor = slot_mapping_npu.to("cpu", non_blocking=True)
    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)

    # key = torch.rand((num_tokens, num_head,head_size_k), dtype=torch.float16)
    key = torch.randint(low=0, high=128, size=(num_tokens, head_size_k), dtype=torch.int8)
    key_npu = key.npu()
    key_expect = cal_slot(key_npu, key_cache_npu, slot_list_np, block_size)

    time.sleep(0.1)
    slot_mapping_list = slot_mapping_cpu.tolist()
    warm_up = 0
    for _ in range(warm_up):
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
            slot_mapping_npu, slot_mapping_list, block_size
        )
        torch.ops._C_ascend.store_kv_block(
            key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size
        )
    N = 101
    for i in range(N):
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
            slot_mapping_npu, slot_mapping_list, block_size
        )
        torch.ops._C_ascend.store_kv_block(
            key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size
        )

    torch.testing.assert_close(key_expect, key_cache_npu, atol=0.001, rtol=0.1)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
