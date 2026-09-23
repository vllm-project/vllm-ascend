# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor orchestration for the GLM-Next CANN KeyPool and PoolKeyIndexer."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.ops.paged_cache import write_pooled_cache

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
        # Load the optional CANN package only when constructing a GLM KPool
        # backend. Use its dispatcher operators, including its Meta kernels.
        try:
            import_module("cann_ops_transformer")
            self._key_pool = torch.ops.cann_ops_transformer.key_pool
            self._pool_key_indexer = torch.ops.cann_ops_transformer.pool_key_indexer
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "GLM-5.3-Flash KPool requires cann_ops_transformer with key_pool and pool_key_indexer, "
                "and the matching CANN ops-transformer operator package."
            ) from exc

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
        if (
            indexer_metadata.cum_query_lens is None
            or indexer_metadata.query_start_loc is None
            or indexer_metadata.start_pos is None
            or indexer_metadata.pool_tail is None
            or indexer_metadata.pooled_key_indices is None
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

        # CANN may save its final tail before another core consumes history.
        # Stage the unfinished pool separately from the new tail, instead of
        # aliasing both through the persistent ring. Pooling is invariant to
        # shifting the starting position by a whole number of pools.
        starts = indexer_metadata.start_pos
        num_reqs = starts.shape[0]
        offsets = torch.arange(index_kpool, device=starts.device)
        start_offsets = starts % index_kpool
        history_positions = starts[:, None] - start_offsets[:, None] + offsets
        tail_pages = tail_metadata.block_table[:num_reqs, 0].long()
        state = tail_cache.new_empty((1 + 2 * num_reqs, index_kpool, 2 * self.head_dim))
        state[1 : num_reqs + 1].copy_(tail_cache[tail_pages[:, None], history_positions % tail_cache.shape[1]])
        history_pages = torch.arange(1, num_reqs + 1, device=starts.device, dtype=torch.int32)
        state_table = (
            (history_pages + num_reqs)[:, None]
            .expand(num_reqs, 1 + (num_tokens + index_kpool - 1) // index_kpool)
            .contiguous()
        )
        state_table[:, 0] = history_pages
        pooled_key = self._key_pool(
            hidden_states,
            key_weight,
            gate_weight,
            compress_ape,
            state,
            state_table,
            start_offsets,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            cu_seqlens=indexer_metadata.query_start_loc,
            cmp_ratio=index_kpool,
            norm_eps=norm_eps,
        )
        if indexer_metadata.retained_tail_indices is not None:
            # Preserve the existing ring's entire rollback window only after
            # KeyPool has consumed its historical rows. Select at most one
            # window per request, so a long prefill never has duplicate stores.
            retained = indexer_metadata.retained_tail_indices.flatten()
            safe_indices = retained.clamp_min(0)
            tail_hidden = hidden_states[safe_indices]
            # Match KeyPool's projection rounding before FP32 normalization;
            # the persistent cache holds normalized K and unmodified gates.
            tail_key = F.linear(tail_hidden, key_weight).float()
            if norm_weight is not None:
                tail_key = F.layer_norm(tail_key, (self.head_dim,), norm_weight, norm_bias, norm_eps)
            tail_gate = F.linear(tail_hidden, gate_weight).float()
            tail_values = torch.cat((tail_key, tail_gate), dim=-1)
            tail_slots = tail_metadata.slot_mapping[safe_indices]
            tail_slots = torch.where(retained >= 0, tail_slots, -1)
            write_pooled_cache(tail_cache.unsqueeze(2), tail_slots, tail_values)
        else:
            # Without speculation, only the unfinished pool remains live.
            query_lens = indexer_metadata.query_start_loc[1:] - indexer_metadata.query_start_loc[:-1]
            ends = start_offsets + query_lens
            output_pages = history_pages + (ends >= index_kpool).int() * num_reqs
            tail_values = state[output_pages.long()]
            tail_positions = starts[:, None] + query_lens[:, None] - (ends % index_kpool)[:, None] + offsets
            tail_slots = tail_pages[:, None] * tail_cache.shape[1] + tail_positions % tail_cache.shape[1]
            valid = (query_lens[:, None] > 0) & (offsets < (ends % index_kpool)[:, None])
            write_pooled_cache(tail_cache.unsqueeze(2), torch.where(valid, tail_slots, -1), tail_values)
        # Only completed pools have valid slots. Gather a fixed number of rows
        # so decode capture/replay never needs nonzero() or a host-side count.
        pooled_rows = pooled_key[indexer_metadata.pooled_key_indices[:num_tokens]]
        write_pooled_cache(
            indexer_cache,
            indexer_metadata.slot_mapping[:num_tokens],
            pooled_rows.to(indexer_cache.dtype),
        )
        # Sharing top-k still advances the compressed cache and raw tail.
        if not compute_topk:
            return None
        assert q_values is not None and weights is not None
        indices, _ = self._pool_key_indexer(
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
