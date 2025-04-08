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

from typing import Optional
import torch

# Implementation of vanilla chunked prefill, should be removed after the kernel is ready for 
# all the corner case
def vanilla_chunked_prefill(
    output: torch.Tensor,
    query: torch.Tensor,  # (num_tokens, num_heads, head_size)
    num_queries_per_kv: int,
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