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
    #
    # Block-aligned variable-length gather: each sequence contributes only the
    # blocks it owns (ceil(seq_len / block_size)), NOT padded to the batch's
    # max sequence length. Max-padding used to materialise a (num_seqs,
    # max_seq_len, num_kv_heads, head_size) dense tensor then discard most of
    # it via a mask, blowing up both peak memory and gather bandwidth ~Nx when
    # a batch mixes short and very long sequences (e.g. a 100K-token decode
    # request alongside short prompts: 24 seqs padded to 109824 tokens each).
    # Here only <=block_size-1 in-block tail padding per sequence remains.
    block_size = key_cache.shape[1]
    num_seqs = len(seq_lens)
    device = key_cache.device
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)

    # blocks owned by each sequence (variable, no max-padding)
    blocks_per_seq = (seq_lens_t + block_size - 1) // block_size  # (num_seqs,)
    max_blocks_needed = int(blocks_per_seq.max().item())
    cols = torch.arange(max_blocks_needed, device=device)
    owned = cols.unsqueeze(0) < blocks_per_seq.unsqueeze(1)  # (num_seqs, max_blocks_needed)
    flat_block_ids = block_table[:num_seqs, :max_blocks_needed].long()[owned]  # (total_blocks,)

    total_tokens = flat_block_ids.shape[0] * block_size
    token_shape = (total_tokens, num_kv_heads, head_size)

    # Trim the <=block_size-1 in-block tail padding of each sequence: map every
    # block-aligned token back to its (sequence, position-in-sequence) so tokens
    # beyond seq_len are masked out.
    tokens_per_seq = blocks_per_seq * block_size  # (num_seqs,)
    token_seq = torch.repeat_interleave(torch.arange(num_seqs, device=device), tokens_per_seq)
    seq_starts = torch.repeat_interleave(
        torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(tokens_per_seq, dim=0)[:-1],
        ]),
        tokens_per_seq,
    )
    pos_in_seq = torch.arange(total_tokens, device=device) - seq_starts
    valid_mask = pos_in_seq < seq_lens_t[token_seq]

    # Gather key and value separately so each per-op gathered intermediate is
    # freed before the next gather: peak is one gathered + one compacted tensor,
    # not both gathered tensors alive at once.
    def _gather_trim(cache: torch.Tensor) -> torch.Tensor:
        gathered = cache.index_select(0, flat_block_ids).reshape(token_shape)
        return gathered[valid_mask].contiguous()

    return _gather_trim(key_cache), _gather_trim(value_cache)
