# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
from typing import Any

import torch
import torch_npu

FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE = 512
SWA_INT_MAX = 2147483647


def npu_large_head_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any,
    *,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    is_prefill_no_cache: bool,
):
    # A2/A3 FIA TND does not support some large head sizes. Keep those prefill
    # cases on an NPU attention op instead of falling back to Python.
    num_tokens = attn_metadata.actual_seq_lengths_q[-1]
    query = query[:num_tokens]
    key, value, actual_seq_lengths_kv = _get_large_head_prefill_kv(
        key,
        value,
        attn_metadata,
        num_tokens,
        key_cache,
        value_cache,
        num_kv_heads,
        head_size,
        is_prefill_no_cache,
    )
    sparse_mode = 3 if attn_metadata.causal else 0
    pre_tokens = SWA_INT_MAX
    next_tokens = 0 if attn_metadata.causal else SWA_INT_MAX
    attn_mask = attn_metadata.attn_mask
    if attn_mask is not None and attn_mask.dtype not in (torch.bool, torch.uint8):
        attn_mask = attn_mask.bool()
    attn_output = torch_npu.npu_fusion_attention(
        query=query,
        key=key,
        value=value,
        head_num=num_heads,
        input_layout="TND",
        atten_mask=attn_mask,
        scale=scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
        actual_seq_kvlen=actual_seq_lengths_kv,
        sparse_mode=sparse_mode,
    )[0]
    return attn_output, None


def _get_large_head_prefill_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any,
    num_tokens: int,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    num_kv_heads: int,
    head_size: int,
    is_prefill_no_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    # PrefillNoCache already has dense TND key/value tensors. Chunked prefill
    # may need historical paged KV cache gathered back to dense TND.
    if is_prefill_no_cache or key_cache is None or value_cache is None:
        return key[:num_tokens], value[:num_tokens], attn_metadata.actual_seq_lengths_q

    seq_lens = attn_metadata.seq_lens_list
    if not seq_lens:
        return key[:num_tokens], value[:num_tokens], attn_metadata.actual_seq_lengths_q

    key, value = _gather_paged_kv_to_dense(
        key_cache,
        value_cache,
        attn_metadata.block_tables,
        seq_lens,
        num_kv_heads,
        head_size,
    )
    actual_seq_lengths_kv = []
    cumsum = 0
    for length in seq_lens:
        cumsum += length
        actual_seq_lengths_kv.append(cumsum)
    return key, value, actual_seq_lengths_kv


def _gather_paged_kv_to_dense(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # npu_fusion_attention consumes dense TND KV, while cached prefill KV is
    # stored by blocks. Gather only valid tokens from the block table.
    block_size = key_cache.shape[1]
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=key_cache.device)
    num_blocks = (max_seq_len + block_size - 1) // block_size
    block_table = block_table[: len(seq_lens), :num_blocks].long()

    flat_block_ids = block_table.reshape(-1)
    max_tokens_padded = num_blocks * block_size
    dense_shape = (len(seq_lens), max_tokens_padded, num_kv_heads, head_size)
    gathered_key = key_cache.index_select(0, flat_block_ids).reshape(dense_shape)
    gathered_value = value_cache.index_select(0, flat_block_ids).reshape(dense_shape)

    positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key_cache.device)
    valid_mask = positions.unsqueeze(0) < seq_lens_tensor.unsqueeze(1)
    return gathered_key[valid_mask].contiguous(), gathered_value[valid_mask].contiguous()
