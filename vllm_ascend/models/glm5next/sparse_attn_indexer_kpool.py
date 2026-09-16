# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor orchestration for the GLM-Next CANN KeyPool and PoolKeyIndexer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_ascend.attention.utils import scatter_paged_cache

if TYPE_CHECKING:
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolTailMetadata,
    )


def append_causal_tail(
    indices: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
) -> None:
    """Append unpooled tokens to the valid prefix required by CANN SFA."""
    tail_width = pool_size - 1
    if tail_width == 0:
        return
    positions = positions.to(torch.int64)
    tail_start = torch.div(positions + 1, pool_size, rounding_mode="floor") * pool_size
    tail_cols = torch.arange(tail_width, device=indices.device, dtype=torch.int64)
    tail_tokens = tail_start.unsqueeze(1) + tail_cols
    tail_values = torch.where(
        tail_cols < (positions + 1 - tail_start).unsqueeze(1),
        tail_tokens,
        -1,
    ).to(indices.dtype)
    # PKI packs complete pools at the front. Short requests have fewer than
    # topk_tokens history entries; placing the tail at that fixed column would
    # leave invalid holes, and SFA would skip the unpooled tokens.
    indices[:, topk_tokens:] = -1
    tail_offsets = tail_start.clamp(max=topk_tokens).unsqueeze(1) + tail_cols
    indices.scatter_(1, tail_offsets, tail_values)


class SparseAttnIndexerKpool(nn.Module):
    """Update KPool caches and optionally select sparse token indices.

    Cache binding and forward-context lookup belong to the model-side backend.
    This helper receives explicit tensors and typed metadata so the cache update
    can be tested independently from the vLLM attention wrapper.
    """

    def __init__(self, topk_tokens: int, head_dim: int) -> None:
        super().__init__()
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor | None,
        weights: torch.Tensor | None,
        positions: torch.Tensor,
        indexer_cache: torch.Tensor,
        tail_cache: torch.Tensor,
        indexer_metadata: AscendIndexerKPoolMetadata,
        tail_metadata: AscendIndexerKPoolTailMetadata,
        *,
        key_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        norm_weight: torch.Tensor | None,
        norm_bias: torch.Tensor | None,
        norm_eps: float,
        compress_ape: torch.Tensor,
        index_kpool: int,
        compute_topk: bool,
    ) -> torch.Tensor | None:
        num_tokens = hidden_states.shape[0]
        if index_kpool <= 0 or self.topk_tokens <= 0 or self.topk_tokens % index_kpool:
            raise ValueError("KPool top-k must be divisible by its positive pool size.")
        if num_tokens == 0:
            return (
                None
                if not compute_topk
                else torch.empty(
                    (0, 1, self.topk_tokens + index_kpool - 1), dtype=torch.int32, device=hidden_states.device
                )
            )
        if any(
            value is None
            for value in (
                indexer_metadata.cum_query_lens,
                indexer_metadata.query_start_loc,
                indexer_metadata.start_pos,
                indexer_metadata.pool_tail,
                indexer_metadata.pooled_key_indices,
            )
        ):
            raise ValueError("GLM KPool metadata requires CANN request boundaries and pooled-key row indices.")
        if indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM KPool compressed cache must be bfloat16.")
        if tail_cache.dtype != torch.float32:
            raise TypeError("GLM KPool compressor tail must be float32.")
        if tail_cache.ndim != 3 or tail_cache.shape[1] < index_kpool or tail_cache.shape[2] != 2 * self.head_dim:
            raise ValueError("GLM KPool tail requires [blocks, capacity, 2 * head_dim] K/gate storage.")
        if tail_metadata.block_size != tail_cache.shape[1]:
            raise ValueError("GLM KPool tail metadata capacity must match the bound cache.")
        if compress_ape.shape != (index_kpool, self.head_dim) or compress_ape.dtype != torch.float32:
            raise ValueError("GLM KPool APE must be FP32 with shape [pool_size, head_dim].")

        if compute_topk and (q_values is None or weights is None):
            raise ValueError("GLM KPool top-k requires query and head weights.")

        pooled_key = torch.ops._C_ascend.npu_key_pool(
            hidden_states,
            key_weight,
            gate_weight,
            compress_ape,
            tail_cache,
            tail_metadata.block_table,
            indexer_metadata.start_pos,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            cu_seqlens=indexer_metadata.query_start_loc,
            cmp_ratio=index_kpool,
            norm_eps=norm_eps,
        )
        # Only completed pools have valid slots. Gather a fixed number of rows
        # so decode capture/replay never needs nonzero() or a host-side count.
        pooled_rows = pooled_key[indexer_metadata.pooled_key_indices[:num_tokens]]
        scatter_paged_cache(
            indexer_cache,
            indexer_metadata.slot_mapping[:num_tokens],
            pooled_rows.to(indexer_cache.dtype),
            indexer_cache.shape[1],
        )
        # Sharing top-k still advances the compressed cache and raw tail.
        if not compute_topk:
            return None
        assert q_values is not None and weights is not None
        indices, _ = torch.ops._C_ascend.npu_pool_key_indexer(
            q_values.to(indexer_cache.dtype),
            indexer_cache,
            weights.to(indexer_cache.dtype),
            indexer_metadata.pool_tail,
            actual_seq_q=indexer_metadata.cum_query_lens,
            actual_seq_k=indexer_metadata.seq_lens,
            block_table=indexer_metadata.block_table,
            layout_q="TND",
            layout_k="PA_BBND",
            topk=self.topk_tokens,
            pool_size=index_kpool,
            mask_mode=3,
        )
        # A2/A3 SFA requires a contiguous valid prefix; the reference indexer
        # puts the running tail at the fixed top-k column for short requests.
        append_causal_tail(indices, positions, self.topk_tokens, index_kpool)
        valid = torch.arange(num_tokens, device=hidden_states.device) < indexer_metadata.cum_query_lens[-1]
        indices.masked_fill_(~valid[:, None], -1)
        return indices.unsqueeze(1)
