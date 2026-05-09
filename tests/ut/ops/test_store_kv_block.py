import random
import time

import numpy as np
import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401

torch.set_printoptions(threshold=np.inf)

# from vllm_ascend.utils import enable_custom_op
# enable_custom_op()


def random_with_zero_prob(zero_prob, index):
    if not 0 <= zero_prob <= 1:
        raise ValueError("the probability must be in [0, 1]")

    if index <= 0:
        raise ValueError("index must be > 0")

    if random.random() < zero_prob:
        return 0

    return random.randint(1, index)


def assert_store_kv_block_registered():
    assert hasattr(torch.ops, "_C_ascend"), "torch.ops._C_ascend is not registered"
    assert hasattr(
        torch.ops._C_ascend,
        "store_kv_block_pre",
    ), "torch.ops._C_ascend.store_kv_block_pre is not registered"
    assert hasattr(
        torch.ops._C_ascend,
        "store_kv_block",
    ), "torch.ops._C_ascend.store_kv_block is not registered"


def golden_store_kv_block(keylist, cache_table, slotmap, block_size):
    expected_cache = cache_table.clone()

    for token_idx, slot in enumerate(slotmap):
        if slot < 0:
            continue

        block_idx = slot // block_size
        block_offset = slot % block_size

        expected_cache[block_idx, block_offset, :, :] = keylist[token_idx, :, :]

    return expected_cache


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [128])
def test_storeKVBlock_with_continuous(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_store_kv_block_registered()

    # keylist = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_tokens, num_head, head_size),
    #     dtype=torch.int8,
    # )
    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    slotmap = []
    for i in range(0, num_tokens):
        slotmap.append(i)

    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    # cache_table = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_blocks, block_size, num_head, head_size),
    #     dtype=torch.int8,
    # )
    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    expected_cache = golden_store_kv_block(
        keylist=keylist,
        cache_table=cache_table,
        slotmap=slotmap,
        block_size=block_size,
    )

    slotmap_np = np.array(slotmap, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slotmap_np).to(torch.int32).npu()

    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    torch.npu.synchronize()

    slot_mapping_list = slot_mapping_cpu.tolist()

    keylist_npu = keylist.npu()
    cache_table_npu = cache_table.clone().npu()

    epoch = 100
    for _ in range(epoch):
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
            slot_mapping_npu,
            slot_mapping_list,
            block_size,
        )
        # group_len_cpu = group_len.cpu()
        # group_key_idx_cpu = group_key_idx.cpu()
        # group_key_cache_idx_cpu = group_key_cache_idx.cpu()
        torch.ops._C_ascend.store_kv_block(
            keylist_npu,
            cache_table_npu,
            group_len,
            group_key_idx,
            group_key_cache_idx,
            block_size,
        )

    torch.testing.assert_close(
        cache_table_npu.cpu(),
        expected_cache,
        rtol=0,
        atol=0,
    )

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [512])
def test_storeKVBlock_without_continuous(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_store_kv_block_registered()

    # keylist = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_tokens, num_head, head_size),
    #     dtype=torch.int8,
    # )
    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    slotmap = []
    r = 0
    for i in range(0, num_tokens):
        r = r + random_with_zero_prob(0.992, 5)
        slotmap.append(i + r)

    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    # cache_table = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_blocks, block_size, num_head, head_size),
    #     dtype=torch.int8,
    # )
    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    # expected_cache = golden_store_kv_block(
    #     keylist=keylist,
    #     cache_table=cache_table,
    #     slotmap=slotmap,
    #     block_size=block_size,
    # )

    slotmap_np = np.array(slotmap, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slotmap_np).to(torch.int32).npu()

    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    torch.npu.synchronize()

    slot_mapping_list = slot_mapping_cpu.tolist()

    keylist_npu = keylist.npu()
    cache_table_npu = cache_table.clone().npu()

    epoch = 100
    start = time.perf_counter()
    for _ in range(epoch):
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
            slot_mapping_npu,
            slot_mapping_list,
            block_size,
        )

        torch.ops._C_ascend.store_kv_block(
            keylist_npu,
            cache_table_npu,
            group_len,
            group_key_idx,
            group_key_cache_idx,
            block_size,
        )

    #     torch.testing.assert_close(
    #         cache_table_npu.cpu(),
    #         expected_cache,
    #         rtol=0,
    #         atol=0,
    #     )
    end = time.perf_counter()
    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [64])
def test_storeKVBlock_without_ascending(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_store_kv_block_registered()

    keylist = torch.randint(
        low=0,
        high=128,
        size=(num_tokens, num_head, head_size),
        dtype=torch.int8,
    )

    slotmap = list(range(num_tokens))
    random.shuffle(slotmap)

    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    cache_table = torch.randint(
        low=0,
        high=128,
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.int8,
    )

    expected_cache = golden_store_kv_block(
        keylist=keylist,
        cache_table=cache_table,
        slotmap=slotmap,
        block_size=block_size,
    )

    slotmap_np = np.array(slotmap, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slotmap_np).to(torch.int32).npu()

    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    torch.npu.synchronize()

    slot_mapping_list = slot_mapping_cpu.tolist()

    keylist_npu = keylist.npu()
    cache_table_npu = cache_table.clone().npu()

    group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
        slot_mapping_npu,
        slot_mapping_list,
        block_size,
    )

    # group_len_cpu = group_len.cpu()
    # group_key_idx_cpu = group_key_idx.cpu()
    # group_key_cache_idx_cpu = group_key_cache_idx.cpu()

    torch.ops._C_ascend.store_kv_block(
        keylist_npu,
        cache_table_npu,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    torch.testing.assert_close(
        cache_table_npu.cpu(),
        expected_cache,
        rtol=0,
        atol=0,
    )

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [32 * 1024])  # 6398
@pytest.mark.parametrize("num_head", [1])  # 512
@pytest.mark.parametrize("block_size", [128])  # 128
@pytest.mark.parametrize("num_blocks", [1773])  # 1599
@pytest.mark.parametrize("head_size", [512])
def test_siso(num_tokens, num_head, block_size, num_blocks, head_size):
    key_cpu = torch.randint(
        low=0,
        high=128,
        size=(num_tokens, num_head, head_size),
        dtype=torch.int8,
    )

    # key_cpu = torch.rand((num_tokens, num_head, head_size), dtype=torch.float16)
    key = key_cpu.npu()

    key_cache_cpu = torch.randint(
        low=0,
        high=128,
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.int8,
    )
    # key_cache_cpu = torch.rand(
    #     (num_blocks, block_size, num_head, head_size),
    #     dtype=torch.float16,
    # )
    key_cache = key_cache_cpu.clone().npu()

    slot_list = []
    r = 0
    for i in range(0, num_tokens):
        r = r + random_with_zero_prob(0.992, 5)
        slot_list.append(i + r)

    assert num_tokens == len(slot_list)

    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = golden_store_kv_block(
        keylist=key_cpu,
        cache_table=key_cache_cpu,
        slotmap=slot_list,
        block_size=block_size,
    )

    epoch = 100
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(epoch):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    torch.npu.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")

    # prof.stop()
    # end = time.perf_counter()
    # avg_ms = (end - start) / N * 1000
    # print(f"python 耗时: {avg_ms:.4f} ms")
    # torch.ops._C_ascend.reshape_and_cache(
    #     key,
    #     value,
    #     key_cache,
    #     value_cache,
    #     slot_mapping,
    # )
    torch.testing.assert_close(key_expect, key_cache.cpu(), atol=0, rtol=0)
    # torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1)


# def cal_slot(key, key_cache, slot_mapping, block_size):
#     key_expect = key_cache.clone()
#     for i, slot in enumerate(slot_mapping):
#         if slot < 0:
#             continue
#         token_key = key[i]
#         block_index = slot // block_size
#         block_offset = slot % block_size
#         key_expect[block_index][block_offset] = token_key
#     return key_expect.npu()


def cal_scatternd(key, key_cache, slot_mapping, block_size):
    key_expect = key_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        key_expect[slot] = token_key

    return key_expect.npu()


# @pytest.mark.parametrize("num_tokens", [16])  # 6398
# @pytest.mark.parametrize("num_head", [1])  # 512
# @pytest.mark.parametrize("block_size", [128])  # 128
# @pytest.mark.parametrize("num_blocks", [1773])  # 1599
# @pytest.mark.parametrize("count", [1])
# def test_siso(num_tokens, num_head, block_size, num_blocks, count):
#     head_size_k = 1
#     key = torch.rand(
#         (num_tokens, num_head, head_size_k),
#         dtype=torch.float16,
#     ).npu()
#     # key = torch.randint(
#     #     low=0,
#     #     high=128,
#     #     size=(num_tokens, head_size_k),
#     #     dtype=torch.int8,
#     # )
#     key_cache = torch.rand(
#         (num_blocks, block_size, num_head, head_size_k),
#         dtype=torch.float16,
#     ).npu()
#     # key_cache = torch.randint(
#     #     low=0,
#     #     high=128,
#     #     size=(num_blocks, block_size, num_head, head_size_k),
#     #     dtype=torch.int8,
#     # )
#
#     slot_list = []
#     for i in range(0, num_tokens):
#         slot_list.append(2 + i)
#     assert num_tokens == len(slot_list)
#     slot_list_np = np.array(slot_list)
#     slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()
#
#     key_expect = cal_slot(key, key_cache, slot_mapping_npu, block_size)
#     warm_up = 0
#
#     for _ in range(warm_up):
#         torch_npu._npu_reshape_and_cache_siso(
#             key,
#             key_cache,
#             slot_mapping_npu,
#         )
#     N = 101
#
#     for _ in range(N):
#         torch_npu._npu_reshape_and_cache_siso(
#             key,
#             key_cache,
#             slot_mapping_npu,
#         )
#
#     torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1)


@pytest.mark.parametrize("num_tokens", [32 * 1024])  # 6398
@pytest.mark.parametrize("num_head", [1])  # 512
@pytest.mark.parametrize("block_size", [128])  # 128
@pytest.mark.parametrize("num_blocks", [1773])  # 1599
@pytest.mark.parametrize("count", [1])
def test_scatter(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k = 64
    key = torch.randint(
        low=0,
        high=128,
        size=(num_tokens, num_head, head_size_k),
        dtype=torch.int8,
    ).npu()
    # key = torch.rand(
    #     (num_tokens, num_head, head_size_k),
    #     dtype=torch.float16,
    # ).npu()

    key_cache = torch.randint(
        low=0,
        high=128,
        size=(num_blocks * block_size, num_head, head_size_k),
        dtype=torch.int8,
    ).npu()
    # key_cache = torch.rand(
    #     (num_blocks * block_size, num_head, head_size_k),
    #     dtype=torch.float16,
    # ).npu()

    slot_list = []
    for i in range(0, num_tokens):
        slot_list.append([2 + i])

    assert num_tokens == len(slot_list)

    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_scatternd(key, key_cache, slot_mapping_npu, block_size)

    N = 101
    for _ in range(N):
        torch_npu.npu_scatter_nd_update_(key_cache, slot_mapping_npu, key)

    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1)


# @pytest.mark.parametrize("num_tokens", [16])  # 6398
# @pytest.mark.parametrize("num_head", [1])  # 512
# @pytest.mark.parametrize("block_size", [128])  # 128
# @pytest.mark.parametrize("num_blocks", [1773])  # 1599
# @pytest.mark.parametrize("count", [1])
# def test_myops(num_tokens, num_head, block_size, num_blocks, count):
#     head_size_k = 64
#     # key_cache = torch.rand(
#     #     (num_blocks, block_size, num_head, head_size_k),
#     #     dtype=torch.float16,
#     # )
#     key_cache = torch.randint(
#         low=0,
#         high=128,
#         size=(num_blocks, block_size, num_head, head_size_k),
#         dtype=torch.int8,
#     )
#     key_cache_npu = key_cache.npu()
#
#     slot_list = []
#     for i in range(0, num_tokens):
#         slot_list.append(2 + i)
#
#     slot_list_np = np.array(slot_list)
#     slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()
#     # slot_mapping_cpu = slot_mapping_npu.to("cpu", non_blocking=True)
#     # num_draft_tensor = slot_mapping_npu.to("cpu", non_blocking=True)
#     slot_mapping_cpu = torch.empty_like(
#         slot_mapping_npu,
#         device="cpu",
#     ).pin_memory()
#     slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
#
#     # key = torch.rand(
#     #     (num_tokens, num_head, head_size_k),
#     #     dtype=torch.float16,
#     # )
#     key = torch.randint(
#         low=0,
#         high=128,
#         size=(num_tokens, head_size_k),
#         dtype=torch.int8,
#     )
#     key_npu = key.npu()
#     key_expect = cal_slot(key_npu, key_cache_npu, slot_list_np, block_size)
#
#     time.sleep(0.1)
#     slot_mapping_list = slot_mapping_cpu.tolist()
#     warm_up = 0
#     for _ in range(warm_up):
#         group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
#             slot_mapping_npu,
#             slot_mapping_list,
#             block_size,
#         )
#         torch.ops._C_ascend.store_kv_block(
#             key_npu,
#             key_cache_npu,
#             group_len,
#             group_key_idx,
#             group_key_cache_idx,
#             block_size,
#         )
#     N = 101
#     for _ in range(N):
#         group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
#             slot_mapping_npu,
#             slot_mapping_list,
#             block_size,
#         )
#         torch.ops._C_ascend.store_kv_block(
#             key_npu,
#             key_cache_npu,
#             group_len,
#             group_key_idx,
#             group_key_cache_idx,
#             block_size,
#         )
#
#     torch.testing.assert_close(
#         key_expect,
#         key_cache_npu,
#         atol=0.001,
#         rtol=0.1,
#     )
#     torch.npu.empty_cache()
#     torch.npu.reset_peak_memory_stats()
