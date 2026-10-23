# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 tensor orchestration for CANN KeyPool and PoolKeyIndexer."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolTailMetadata,
    )


class SparseAttnIndexerKpool(nn.Module):
    """Update KPool caches and optionally select sparse token indices."""

    def __init__(self, topk_tokens: int, head_dim: int) -> None:
        super().__init__()
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        try:
            import_module("cann_ops_transformer")
            adapters = import_module("vllm_ascend.ops.cann_keypool")
            fused = import_module("vllm_ascend.ops.triton.glm5_next_cann_keypool")
            self._gather_retained_hidden = adapters.gather_retained_hidden
            self._pack_keypool_indices = fused.pack_keypool_indices
            self._prepare_keypool_state = adapters.prepare_keypool_state
            self._quantize_keypool_query = adapters.quantize_keypool_query
            self._write_keypool_caches = fused.write_keypool_caches
            self._key_pool = torch.ops.cann_ops_transformer.key_pool
            self._pool_key_indexer = torch.ops.cann_ops_transformer.pool_key_indexer
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "GLM-5.3-Flash KPool requires cann_ops_transformer with key_pool and pool_key_indexer, "
                "the matching CANN ops-transformer operator package and Triton cache/index adapters."
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
        output_buffer: torch.Tensor | None = None,
        indexer_scale_cache: torch.Tensor | None = None,
        skip_empty_cp_scoring: bool = True,
    ) -> torch.Tensor | None:
        query_metadata = indexer_metadata.query_metadata
        num_tokens = hidden_states.shape[0] if query_metadata is None else query_metadata.positions.shape[0]
        output_width = self.topk_tokens + index_kpool - 1
        if compute_topk and output_buffer is not None:
            if (
                output_buffer.ndim != 2
                or output_buffer.dtype != torch.int32
                or output_buffer.device != hidden_states.device
                or output_buffer.stride(1) != 1
                or (num_tokens > 1 and output_buffer.stride(0) < output_buffer.shape[1])
            ):
                raise ValueError("GLM KPool output buffer requires int32 contiguous columns on the query device.")
            if output_buffer.shape[0] < num_tokens or output_buffer.shape[1] < output_width:
                raise ValueError("GLM KPool output buffer must cover the token rows and top-k width.")
        if index_kpool <= 0 or self.topk_tokens <= 0 or self.topk_tokens % index_kpool:
            raise ValueError("KPool top-k must be divisible by its positive pool size.")
        if hidden_states.shape[0] == 0:
            if not compute_topk:
                return None
            if output_buffer is not None:
                return output_buffer[:0, :output_width].unsqueeze(1)
            return torch.empty((0, 1, output_width), dtype=torch.int32, device=hidden_states.device)
        if (
            indexer_metadata.cum_query_lens is None
            or indexer_metadata.query_start_loc is None
            or indexer_metadata.start_pos is None
            or indexer_metadata.pool_tail is None
            or indexer_metadata.pooled_slot_mapping is None
            or tail_metadata.cann_inputs is None
        ):
            raise ValueError("GLM KPool metadata requires CANN request boundaries and pooled-key row indices.")
        if indexer_cache.dtype != torch.float8_e4m3fn:
            raise TypeError("GLM FP8 CANN indexer requires float8_e4m3fn keys.")
        if indexer_scale_cache is None or indexer_scale_cache.dtype != torch.float32:
            raise ValueError("GLM KPool FP8 cache requires FP32 per-pool scales.")
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

        # Keep history separate: CANN can save the tail before other cores
        # finish reading history. Whole-pool shifts preserve the pooling result.
        state = self._prepare_keypool_state(tail_cache, indexer_metadata.start_pos, tail_metadata, index_kpool)
        pooled_key = self._key_pool(
            hidden_states,
            key_weight,
            gate_weight,
            compress_ape,
            state,
            indexer_metadata.keypool_state_table,
            indexer_metadata.keypool_start_offsets,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            cu_seqlens=indexer_metadata.query_start_loc,
            cmp_ratio=index_kpool,
            norm_eps=norm_eps,
        )
        tail_key = tail_gate = None
        if tail_metadata.cann_inputs.retained_indices is not None:
            # Re-project the live MTP rollback window missing from CANN state.
            tail_hidden = self._gather_retained_hidden(hidden_states, tail_metadata.cann_inputs.retained_indices)
            tail_key = F.linear(tail_hidden, key_weight).float()
            if norm_weight is not None:
                tail_key = F.layer_norm(tail_key, (self.head_dim,), norm_weight, norm_bias, norm_eps)
            tail_gate = F.linear(tail_hidden, gate_weight).float()
        self._write_keypool_caches(
            tail_cache,
            indexer_cache,
            tail_key,
            tail_gate,
            pooled_key,
            indexer_metadata,
            tail_metadata,
            state,
            scale_cache=indexer_scale_cache,
        )
        # Sharing top-k still advances the compressed cache and raw tail.
        if not compute_topk:
            return None
        assert q_values is not None and weights is not None
        score_metadata = indexer_metadata
        if query_metadata is not None:
            if query_metadata.seq_lens is None or query_metadata.pool_tail is None:
                raise ValueError("CANN KPool CP requires rank-local causal key lengths and pool tails.")
            if q_values.shape[0] != num_tokens or weights.shape[0] != num_tokens:
                raise ValueError("CANN KPool CP query tensors must match the local query metadata.")
            score_metadata = replace(
                indexer_metadata,
                positions=query_metadata.positions,
                cum_query_lens=query_metadata.cum_query_lens.to(indexer_metadata.cum_query_lens.dtype),
                seq_lens=query_metadata.seq_lens,
                pool_tail=query_metadata.pool_tail,
                num_actual_tokens=query_metadata.num_actual_tokens,
            )
        if num_tokens == 0:
            if output_buffer is not None:
                return output_buffer[:0, :output_width].unsqueeze(1)
            return torch.empty((0, 1, output_width), dtype=torch.int32, device=hidden_states.device)
        # DP padding can leave an entire CP rank without actual queries.
        # Global KPool cache updates above must still run on every TP rank.
        # PoolKeyIndexer must not receive a nonempty query allocation with
        # all-zero actual_seq_q/actual_seq_k. Keep padded result rows invalid.
        # FULL graph capture must retain the operator for changing replay data.
        if skip_empty_cp_scoring and query_metadata is not None and query_metadata.num_actual_tokens == 0:
            if output_buffer is None:
                output = torch.empty((num_tokens, output_width), dtype=torch.int32, device=hidden_states.device)
            else:
                output = output_buffer[:num_tokens, :output_width]
            output.fill_(-1)
            return output.unsqueeze(1)
        query, q_descale = self._quantize_keypool_query(q_values)
        quant_kwargs = {"q_descale": q_descale, "k_descale": indexer_scale_cache[..., 0], "quant_mode": 0}
        indices, _ = self._pool_key_indexer(
            query,
            indexer_cache,
            weights.to(torch.bfloat16),
            score_metadata.pool_tail,
            actual_seq_q=score_metadata.cum_query_lens,
            actual_seq_k=score_metadata.seq_lens,
            block_table=indexer_metadata.block_table,
            layout_q="TND",
            layout_k="PA_BBND",
            topk=self.topk_tokens,
            pool_size=index_kpool,
            mask_mode=3,
            **quant_kwargs,
        )
        return self._pack_keypool_indices(indices, score_metadata, self.topk_tokens, index_kpool, output_buffer)
