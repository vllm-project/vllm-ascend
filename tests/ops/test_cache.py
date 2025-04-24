# SPDX-License-Identifier: Apache-2.0
# forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_cache.py
# test https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/mla/common.py#L1399-L1406
# test https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/mla/common.py#L1221-L1228
import random
from typing import Optional

import pytest
import torch
#from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]
SEEDS = [0]


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(num_blocks,
                       block_size,
                       entry_size,
                       dtype=cache_dtype,
                       device=device)


def concat_and_cache_mla_torch(
        kv_c_normed: torch.Tensor,  # [num_tokens, num_kv_head, nope]
        k_pe: torch.Tensor,  # [num_tokens, num_kv_head, rope]
        kv_cache: torch.
    Tensor,  # [num_blocks, block_size, num_kv_head, nope + rope]
        slot_mapping,  # [num_tokens]
):
    num_blocks = kv_cache.size()[0]
    block_size = kv_cache.size()[1]
    num_kv_head = k_pe.size()[1]

    idx_for_copy = slot_mapping // block_size * block_size + slot_mapping % block_size
    kv_cache = kv_cache.view(num_blocks * block_size, num_kv_head, -1)
    kv_cache[idx_for_copy] = torch.cat([kv_c_normed.unsqueeze(1), k_pe],
                                       dim=-1)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
#@pytest.mark.cpu_model
#@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    device = "npu"
    kv_cache_dtype = "auto"
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens,
                       qk_rope_head_dim,
                       dtype=dtype,
                       device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    # scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                 kv_cache_dtype, device)
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    ref_kv_cache = ref_temp

    #ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
    #kv_cache_dtype, scale)

    concat_and_cache_mla_torch(kv_c, k_pe.unsqueeze(1), kv_cache, slot_mapping)
    torch.testing.assert_close(kv_cache, ref_kv_cache)


def _fill_mla_cache(cache: torch.Tensor, kv_cache_dtype: str):
    rand_dtype = torch.float16 if kv_cache_dtype == "fp8" else cache.dtype
    vals = torch.randn(*cache.shape, device=cache.device, dtype=rand_dtype)
    if kv_cache_dtype == "fp8":
        temp = torch.zeros_like(cache)
        ops.convert_fp8(temp, vals, 1.0, kv_dtype=kv_cache_dtype)
        vals = temp
    cache.copy_(vals)


def gather_cache_torch(
        src_cache: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
        dst: torch.Tensor,  # [TOT_TOKENS, ENTRIES]
        block_table: torch.Tensor,  # [BATCH, BLOCK_INDICES]
        cu_seq_lens: torch.Tensor,  # [BATCH+1]
        batch_size: int,
        seq_starts: Optional[torch.Tensor] = None  # Optional: [BATCH]
) -> None:
    """
        从源缓存中收集序列数据到目标tensor
        Args:
        src_cache: 源缓存tensor [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
        dst: 目标tensor [TOT_TOKENS, ENTRIES]
        block_table: 块表映射 [BATCH, BLOCK_INDICES]
        cu_seq_lens: 累积序列长度 [BATCH+1]
        batch_size: 批大小
        seq_starts: 可选,每个batch的起始偏移 [BATCH]
        """
    # 基本参数检查
    assert src_cache.dtype == dst.dtype, "src_cache and dst must have same dtype"
    assert block_table.dtype == torch.int32, "block_table must be int32"
    assert cu_seq_lens.dtype == torch.int32, "cu_seq_lens must be int32"

    if seq_starts is not None:
        assert seq_starts.dtype == torch.int32, "seq_starts must be int32"

    block_size = src_cache.size(1)
    # 对每个batch进行处理
    for bid in range(batch_size):
        # 获取当前batch的序列起始和结束位置
        seq_start = cu_seq_lens[bid].item()
        seq_end = cu_seq_lens[bid + 1].item()
        seq_len = seq_end - seq_start

        if seq_len == 0:
            continue

        # 计算需要的block数
        tot_blocks = (seq_len + block_size - 1) // block_size

        # 如果有seq_starts,计算block偏移
        offset = 0
        if seq_starts is not None:
            offset = seq_starts[bid].item() // block_size

        # 获取当前batch的block table
        batch_block_table = block_table[bid, offset:offset + tot_blocks]
        # 计算完整blocks和最后一个partial block
        full_blocks = tot_blocks - 1 if seq_len % block_size else tot_blocks
        partial_block_size = seq_len % block_size if seq_len % block_size else 0
        # 复制完整blocks
        dst_start = seq_start
        for i in range(full_blocks):
            block_id = batch_block_table[i].item()
            # 复制整个block，移除HEAD维度
            dst[dst_start:dst_start +
                block_size] = src_cache[block_id].squeeze(1)
            dst_start += block_size

        # 处理最后一个不完整block
        if partial_block_size > 0:
            block_id = batch_block_table[full_blocks].item()
            dst[dst_start:dst_start + partial_block_size] = \
            src_cache[block_id, :partial_block_size].squeeze(1)


@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1024])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype",
                         ["auto"])  # You can also test "fp8" if needed.
@torch.inference_mode()
def test_gather_cache_mla(kv_lora_rank, qk_rope_head_dim, block_size,
                          num_blocks, max_seq_len, batch_size, dtype,
                          kv_cache_dtype):
    device = "npu"
    entry_size = kv_lora_rank + qk_rope_head_dim
    src_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                  kv_cache_dtype, device)
    _fill_mla_cache(src_cache, kv_cache_dtype=kv_cache_dtype)
    seq_len_tensor = torch.randint(0,
                                   max_seq_len + 1, (batch_size, ),
                                   device=device)

    total_tokens = seq_len_tensor.sum()
    cu_seq_lens = torch.empty((batch_size + 1),
                              dtype=torch.int32,
                              device=device)
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)

    # tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
    block_table = torch.empty((batch_size, num_blocks),
                              dtype=torch.int32,
                              device=device)

    for b in range(batch_size):
        perm = torch.randperm(num_blocks, device=device)
        block_table[b, :] = perm

    expected = torch.zeros((total_tokens, entry_size),
                           dtype=src_cache.dtype,
                           device=device)
    # 修正的 seq_starts 生成
    max_start = max_seq_len // 2
    seq_starts = torch.randint(0,
                               max_start + 1, (batch_size, ),
                               dtype=torch.int32,
                               device=device)
    '''
    expected_batches = []
    for b in range(batch_size):
        s = seq_len_tensor[b]
        if s == 0:
            continue
        tot = tot_blocks_tensor[b]
        
        # 如果使用 seq_starts，需要考虑偏移量
        offset = seq_starts[b] // block_size  # 如果需要考虑 seq_starts 的偏移
        blocks = block_table[b, offset:offset + tot].tolist()
        #blocks = block_table[b, :tot].tolist()  # 当 seq_starts 为 0 时保持原样
        gathered_rows = []
        for i in range(tot - 1):
            gathered_rows.append(src_cache[blocks[i]])
        remaining = s - (tot - 1) * block_size
        gathered_rows.append(src_cache[blocks[-1], :remaining, :])
        batch_expected = torch.cat(gathered_rows, dim=0)
        expected_batches.append(batch_expected)
    expected = torch.cat(expected_batches, dim=0)
    '''

    #gather_cache_torch(src_cache, expected, block_table, cu_seq_lens, batch_size, seq_starts)
    gather_cache_torch(src_cache, expected, block_table, cu_seq_lens,
                       batch_size, seq_starts)
    '''
    dst = torch.zeros((total_tokens, entry_size),
            dtype=src_cache.dtype,
            device=device)
    ops.gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts)
    torch.testing.assert_close(dst, expected)
    '''
