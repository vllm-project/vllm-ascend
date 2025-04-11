# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List
import torch
from vllm.model_executor.layers.linear import ColumnParallelLinear
# Implementation of vanilla chunked prefill, should be removed after the kernel is ready for 
# all the corner case
def vanilla_chunked_prefill(
    output: torch.Tensor,
    query: torch.Tensor,  # (num_tokens, num_heads, head_size)
    key_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size)
    value_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size,)
    block_tables: torch.Tensor,  # (num_seqs, max_num_blocks_per_seq)
    cu_seqlen_q: torch.Tensor,  # (num_seqs + 1,)
    cu_seqlen_k: torch.Tensor,  # (num_seqs + 1,)
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool = True,
) -> None:
    num_query_heads = query.shape[1]
    head_dim = value_cache.shape[3]
    num_kv_heads = value_cache.shape[2]
    block_size = value_cache.shape[1]
    num_batch = cu_seqlen_q.shape[0] - 1
    num_tokens = query.shape[0]
    max_num_blocks_per_seq = block_tables.shape[1]

    key = key_cache[block_tables].view(
        num_batch, max_num_blocks_per_seq * block_size, num_kv_heads, head_dim
    )

    value = value_cache[block_tables].view(
        num_batch, max_num_blocks_per_seq * block_size, num_kv_heads, head_dim
    )
    key = key[:, :max_seqlen_k, :, :]
    value = value[:, :max_seqlen_k, :, :]

    seqlen_k = cu_seqlen_k[1:] - cu_seqlen_k[:-1]
    seqlen_q = cu_seqlen_q[1:] - cu_seqlen_q[:-1]
    seqlen_q = seqlen_q.view(-1, 1)
    seqlen_k = seqlen_k.view(-1, 1)
    seqlen_diff = seqlen_k - seqlen_q
    q_idx_mask = (
        torch.arange(0, max_seqlen_q, device="npu").view(1, -1).repeat(num_batch, 1)
    )
    k_idx_mask = (
        torch.arange(0, max_seqlen_k, device="npu").view(1, -1).repeat(num_batch, 1)
    )
    q_mask = q_idx_mask < seqlen_q
    k_mask = k_idx_mask < seqlen_k

    # calculate idx for causal mask of query    [batch, max_seqlen_q]
    causal_mask_idx = (q_idx_mask + seqlen_diff)[q_mask]

    # generate causal mask [batch, max_seqlen_q, max_seqlen_k]
    tril_mask = torch.tril(torch.ones(max_seqlen_k, max_seqlen_k, device="npu"))
    tril_mask[tril_mask == 0] = float("-inf")
    tril_mask[tril_mask == 1] = 0
    causal_mask = tril_mask[causal_mask_idx]
    causal_mask_padding = torch.empty(
        [num_batch, max_seqlen_q, max_seqlen_k], device="npu"
    ).fill_(float("-inf"))
    causal_mask_padding[q_mask] = causal_mask
    # to [batch, num_heads, max_seqlen_q, max_seqlen_k]
    causal_mask_padding = causal_mask_padding.unsqueeze(1)

    pad_q = torch.zeros(
        [num_batch, max_seqlen_q, num_query_heads, head_dim],
        device="npu",
        dtype=query.dtype,
    )
    pad_k = torch.zeros(
        [num_batch, max_seqlen_k, num_kv_heads, head_dim],
        device="npu",
        dtype=key.dtype,
    )
    pad_v = torch.zeros(
        [num_batch, max_seqlen_k, num_kv_heads, head_dim],
        device="npu",
        dtype=value.dtype,
    )
    pad_q[q_mask] = query
    pad_k[k_mask] = key[k_mask]
    pad_v[k_mask] = value[k_mask]

    if num_query_heads > num_kv_heads:
        pad_k = pad_k.view([num_batch, max_seqlen_k, num_kv_heads, 1, head_dim])
        pad_k = pad_k.repeat(1, 1, 1, num_query_heads // num_kv_heads, 1).view(
            [num_batch, max_seqlen_k, num_query_heads, head_dim]
        )
        pad_v = pad_v.view([num_batch, max_seqlen_k, num_kv_heads, 1, head_dim])
        pad_v = pad_v.repeat(1, 1, 1, num_query_heads // num_kv_heads, 1).view(
            [num_batch, max_seqlen_k, num_query_heads, head_dim]
        )
    # permute to [b, h, n, k]
    pad_q = pad_q.permute(0, 2, 1, 3)
    pad_k = pad_k.permute(0, 2, 1, 3)
    pad_v = pad_v.permute(0, 2, 1, 3)
    attn_mask = torch.empty([num_batch, 1, 1, max_seqlen_k], device="npu").fill_(
        float("-inf")
    )
    attn_mask[:, :, :, :max_seqlen_k].masked_fill_(k_mask[:, None, None, :], 0)
    # [b, h, f, t]
    attn_weights = torch.einsum("bhqd,bhkd->bhqk", pad_q, pad_k)
    attn_weights *= scale
    attn_mask = attn_mask.float()
    attn_weights = attn_weights + attn_mask
    if causal:
        attn_weights = attn_weights + causal_mask_padding

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, pad_v.float())
    attn_output = attn_output.permute(0, 2, 1, 3)

    attn_output = (
        attn_output[q_mask].view([-1, num_query_heads, head_dim]).to(output.dtype)
    )
    output.copy_(attn_output)
    return attn_output


def vanilla_chunked_prefill_mla(
    output: torch.Tensor,   # (num_tokens, num_heads, v_head_dim)
    query: torch.Tensor,  # (num_tokens, num_heads, nope_dim + rope_dim)
    kv_cache: torch.Tensor,  # (num_blocks, block_size, latent_kv)
    block_tables: torch.Tensor,  # (batch_size, max_num_blocks_per_seq)
    query_lens: torch.Tensor,   # (batch_size)
    context_lens: torch.Tensor, # (batch_size)
    kv_b_proj: ColumnParallelLinear,    # ()
    max_query_len: int,
    max_context_len: int,
    nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool = True
) -> None:
    batch_size = block_tables.size(0)
    assert query_lens.size(0) == batch_size
    num_tokens = query.size(0)
    num_heads = query.size(1)
    num_blocks = kv_cache.size(0)
    block_size = kv_cache.size(1)
    latent_kv_dim = kv_cache.size(3) - rope_dim
    max_num_blocks_per_seq = block_tables.size(1)
    batch_size = query_lens.size(0)
    kv_cache = kv_cache.squeeze()
    # select kv_c out as [batch_size, max_context_len, latent_kv + rope_dim]
    cache_kv_c_pe = kv_cache[block_tables].view(batch_size, max_num_blocks_per_seq * block_size, latent_kv_dim + rope_dim)[:, :max_context_len, :]
    # get kv_c and k_pe
    # cached_kv_c: [batch_size, max_context_len, latent_kv]
    # cached_k_pe: [batch_size, max_context_len, rope_dim]
    cache_kv_c = cache_kv_c_pe[:, :, :latent_kv_dim]
    cache_k_pe = cache_kv_c_pe[:, :, latent_kv_dim:]
    # get k_rope and v
    # k_nope: [batch_size, max_context_len, num_heads, nope_dim]
    # value:  [batch_size, max_context_len, num_heads, v_head_dim]
    k_nope, value = kv_b_proj(cache_kv_c)[0].view(batch_size, max_context_len, num_heads, nope_dim + v_head_dim).split([nope_dim, v_head_dim], dim=-1)
    # key:    [batch_size, max_context_len, num_hads, rope_dim + nope_dim]
    key = torch.cat([k_nope, cache_k_pe.unsqueeze(2).expand(-1, -1, num_heads, -1)], dim=-1)

    context_lens = context_lens.view(-1, 1).to("npu")
    query_lens = query_lens.view(-1, 1).to("npu")
    seq_diff = context_lens - query_lens

    q_idx_mask = (
        torch.arange(0, max_query_len, device="npu").view(1, -1).repeat(batch_size, 1)
    )
    kv_c_idx_mask = (
        torch.arange(0, max_context_len, device="npu").view(1, -1).repeat(batch_size, 1)
    )
    kv_c_mask = kv_c_idx_mask < context_lens
    q_mask = q_idx_mask < query_lens

    # calculate idx for causal mask of query    [batch, max_seqlen_q]
    causal_mask_idx = (q_idx_mask + seq_diff)[q_mask]

    # generate causal mask [batch, max_seqlen_q, max_seqlen_k]
    tril_mask = torch.tril(torch.ones(max_context_len, max_context_len, device="npu"))
    tril_mask[tril_mask == 0] = float("-inf")
    tril_mask[tril_mask == 1] = 0
    causal_mask = tril_mask[causal_mask_idx]
    causal_mask_padding = torch.empty(
        [batch_size, max_query_len, max_context_len], device="npu"
    ).fill_(float("-inf"))
    causal_mask_padding[q_mask] = causal_mask
    # to [batch, num_heads, max_seqlen_q, max_seqlen_k]
    causal_mask_padding = causal_mask_padding.unsqueeze(1)

    pad_q = torch.zeros(
        [batch_size, max_query_len, num_heads, rope_dim + nope_dim],
        device="npu",
        dtype=query.dtype,
    )
    pad_k = torch.zeros(
        [batch_size, max_context_len, num_heads, rope_dim + nope_dim],
        device="npu",
        dtype=key.dtype,
    )
    pad_v = torch.zeros(
        [batch_size, max_context_len, num_heads, v_head_dim],
        device="npu",
        dtype=value.dtype,
    )
    pad_q[q_mask] = query
    pad_k[kv_c_mask] = key[kv_c_mask]
    pad_v[kv_c_mask] = value[kv_c_mask]

    pad_q = pad_q.permute(0, 2, 1, 3)
    pad_k = pad_k.permute(0, 2, 1, 3)
    pad_v = pad_v.permute(0, 2, 1, 3)
    attn_mask = torch.empty([batch_size, 1, 1, max_context_len], device="npu").fill_(
        float("-inf")
    )
    attn_mask[:, :, :, :max_context_len].masked_fill_(kv_c_mask[:, None, None, :], 0)
    # [b, h, f, t]
    attn_weights = torch.einsum("bhqd,bhkd->bhqk", pad_q, pad_k)
    attn_weights *= scale
    attn_mask = attn_mask.float()
    attn_weights = attn_weights + attn_mask
    if causal:
        attn_weights = attn_weights + causal_mask_padding

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, pad_v.float())
    attn_output = attn_output.permute(0, 2, 1, 3)

    attn_output = (
        attn_output[q_mask].view([-1, num_heads, v_head_dim]).to(output.dtype)
    )
    output.copy_(attn_output)
    return attn_output

def vanilla_decode_mla(
    query: torch.Tensor,      # [num_tokens, num_heads, latent_dim + rope_dim]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, latent_dim + rope_dim]
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    block_table: torch.Tensor,  # [batch_size, max_block_size]
    context_lens: List[int],
    mla_vhead_size: int,
    rope_dim: int,
    output: torch.Tensor
):
    batch_size = block_table.size()[0]
    max_block_size = block_table.size()[1]
    reduce_dim = key_cache.size()[-1]
    block_size = key_cache.size()[1]
    latent_dim = reduce_dim - rope_dim
    num_tokens = query.size()[0]
    head_ratio = num_heads // num_kv_heads
    kv_c_and_pe = key_cache[block_table].view([batch_size, max_block_size * block_size, num_kv_heads, reduce_dim])
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, device="npu").view(batch_size, 1)
    # [batch_size, max_context_len, num_kv_heads, latent_dim + rope_dim]
    # since the kv head is 1 in deepseek, we use expand here for perf
    kv_c_and_pe = kv_c_and_pe[:, :max_context_len, :, :].expand(-1, -1, num_heads, 1)
    kv_c = kv_c_and_pe[..., :latent_dim]
    kv_idx_mask = (
        torch.arange(0, max_context_len, device="npu").view(1, -1).repeat(batch_size, 1)
    )
    # [batch_size, max_context_len]
    kv_idx_mask = kv_idx_mask < context_lens
    query = query.unsqueeze(1)
    attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, kv_c_and_pe)
    attn_weights *= scale
    attn_weights = attn_weights + kv_idx_mask[:, -1, -1, :].float()
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, kv_c.float()).view(-1, num_heads, latent_dim)
    output.copy_(attn_output)
    return output






    

