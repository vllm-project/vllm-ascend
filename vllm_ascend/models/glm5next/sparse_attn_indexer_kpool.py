# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor orchestration for the GLM-Next Triton KPool indexer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_ascend.ops.triton.glm5_next_kpool_state_compress import (  # type: ignore[import-untyped]
    glm5_next_kpool_state_compress_and_write_cache_triton,
)
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (  # type: ignore[import-untyped]
    glm5_next_lightning_indexer_triton,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolStateMetadata,
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
        k: torch.Tensor,
        q_values: torch.Tensor | None,
        weights: torch.Tensor | None,
        positions: torch.Tensor,
        indexer_cache: torch.Tensor,
        state_cache: torch.Tensor,
        indexer_metadata: AscendIndexerKPoolMetadata,
        state_metadata: AscendIndexerKPoolStateMetadata,
        *,
        gate_score: torch.Tensor,
        compress_ape: torch.Tensor,
        index_kpool: int,
        max_pool_seq_len: int,
        compute_topk: bool,
    ) -> torch.Tensor | None:
        num_tokens = k.shape[0]
        if index_kpool <= 0 or self.topk_tokens % index_kpool:
            raise ValueError("KPool top-k must be divisible by its positive pool size.")
        if num_tokens == 0:
            return (
                None
                if not compute_topk
                else torch.empty((0, 1, self.topk_tokens + index_kpool - 1), dtype=torch.int32, device=k.device)
            )
        if indexer_metadata.cum_query_lens is None or indexer_metadata.raw_seq_lens is None:
            raise ValueError("GLM KPool metadata requires cum_query_lens and raw_seq_lens.")
        if indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM KPool compressed cache must be bfloat16.")
        if state_cache.dtype != torch.float32 or k.dtype != torch.float32 or gate_score.dtype != torch.float32:
            raise TypeError("GLM KPool keys, gates and compressor state must be float32.")
        if state_cache.ndim == 4:
            state_cache = state_cache.view(state_cache.shape[0], state_cache.shape[1], -1)
        if state_cache.shape[-1] != 2 * self.head_dim or gate_score.shape != k.shape:
            raise ValueError("GLM KPool state stores one key and gate vector per token.")
        if compress_ape.shape != (index_kpool, self.head_dim) or compress_ape.dtype != torch.float32:
            raise ValueError("GLM KPool APE must be FP32 with shape [pool_size, head_dim].")

        glm5_next_kpool_state_compress_and_write_cache_triton(
            state_cache,
            indexer_cache,
            k,
            gate_score,
            compress_ape,
            positions,
            indexer_metadata.cum_query_lens,
            indexer_metadata.raw_seq_lens,
            state_metadata.slot_mapping[:num_tokens],
            state_metadata.block_table,
            indexer_metadata.slot_mapping[:num_tokens],
            index_kpool,
        )
        # Sharing top-k still advances both state caches.
        if not compute_topk:
            return None
        if q_values is None or weights is None:
            raise ValueError("GLM KPool top-k requires query and head weights.")
        indices = glm5_next_lightning_indexer_triton(
            q_values,
            indexer_cache,
            weights.to(q_values.dtype),
            indexer_metadata.cum_query_lens,
            indexer_metadata.seq_lens,
            indexer_metadata.block_table,
            positions,
            index_topk=self.topk_tokens,
            index_kpool=index_kpool,
            max_pool_seq_len=max_pool_seq_len,
        )
        # A2/A3 SFA requires a contiguous valid prefix; the reference indexer
        # puts the running tail at the fixed top-k column for short requests.
        append_causal_tail(indices[:, 0], positions, self.topk_tokens, index_kpool)
        valid = torch.arange(num_tokens, device=k.device) < indexer_metadata.cum_query_lens[-1]
        indices.masked_fill_(~valid[:, None, None], -1)
        return indices
