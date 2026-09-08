# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend orchestrator for the GLM-5 kpool indexer.

Assembles the pure-PyTorch components (paged-cache scatter/gather, the
triton kpool compress kernel, the top-k scorers) behind the
``SparseAttnIndexerKpool`` call signature used by the NVIDIA backend, so
the GLM NoPE layers run their trained sparse attention on NPU:

    state scatter -> pool assembly -> kpool compress+write (triton)
    -> topk select (pytorch; the CANN pool_key_indexer wheel is not
       available on CANN 9.1.0)
"""

from __future__ import annotations

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_to_tokens

from vllm_ascend.ops.glm5_next_kpool_compress import (
    glm5_next_kpool_compress_and_write_cache,
)
from vllm_ascend.ops.indexer_kpool_topk import (
    gather_compressor_state,
    indexer_kpool_topk_pytorch,
    indexer_kpool_topk_static,
    scatter_paged_cache,
)

logger = init_logger(__name__)


def bound_cache(layer) -> torch.Tensor:
    """Resolve the current virtual engine's bound cache tensor."""
    context = get_forward_context()
    virtual_engine = getattr(context, "virtual_engine", 0) or 0
    cache = layer.kv_cache
    if isinstance(cache, (list, tuple)):
        cache = cache[virtual_engine]
    if isinstance(cache, (list, tuple)):
        if len(cache) == 1:
            cache = cache[0]
        elif all(isinstance(tensor, torch.Tensor) for tensor in cache):
            return tuple(cache)
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"GLM-5 Indexer cache {type(layer).__name__} is not bound.")
    return cache


class AscendGlm5KpoolIndexerOp:
    """NPU kpool indexer with the ``SparseAttnIndexerKpool`` call signature.

    Expected cache layouts:
      * indexer K cache: ``[blocks, block, 1, head_dim]`` bf16, pool-granular
        slots (one row per completed kpool pool)
      * compressor state cache: ``[blocks, block, 2*head_dim]`` bf16,
        token-granular slots holding ``[K, gate]``
    """

    def __init__(
        self,
        k_cache_layer,
        state_cache_layer,
        topk_tokens: int,
        head_dim: int,
        topk_indices_buffer: torch.Tensor | None,
        attn_layer_name: str,
    ) -> None:
        self.k_cache = k_cache_layer
        self.state_cache = state_cache_layer
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.topk_indices_buffer = topk_indices_buffer
        self.attn_layer_name = attn_layer_name

    def __call__(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Unused: part of the SparseAttnIndexerKpool call signature.
        del hidden_states
        if gate_score is None or compress_ape is None or positions is None:
            raise ValueError("GLM-5 kpool requires gate_score, compress_ape, and positions.")

        context = get_forward_context()
        metadata = context.attn_metadata
        if not isinstance(metadata, dict):
            raise TypeError("GLM-5 Indexer requires per-layer attention metadata.")
        state_metadata = metadata[self.state_cache.prefix]
        indexer_metadata = metadata[self.k_cache.prefix]
        attn_metadata = metadata[self.attn_layer_name]
        state_cache = bound_cache(self.state_cache)
        indexer_cache = bound_cache(self.k_cache)
        if not isinstance(state_cache, torch.Tensor):
            raise TypeError("GLM-5 compressor state cache must be one tensor.")
        if not isinstance(indexer_cache, torch.Tensor) or indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM-5 indexer cache must be one bfloat16 K tensor.")

        # The vLLM allocator binds these caches as [blocks, num_heads, block,
        # head_size] (heads=1); the addressing below assumes
        # [blocks, block, 1, head_size]. Normalize with a permute view so the
        # scatter/gather/compress math below addresses the intended layout.
        def _as_bbd(cache: torch.Tensor) -> torch.Tensor:
            if cache.ndim == 4 and cache.shape[1] == 1 and cache.shape[2] != 1:
                return cache.permute(0, 2, 1, 3)
            return cache

        state_cache = _as_bbd(state_cache)
        indexer_cache = _as_bbd(indexer_cache)

        # FULL-graph mode: every captured shape must be static, so derive
        # num_tokens from the (padded, fixed) positions tensor and select ALL
        # tokens instead of the data-dependent nonzero() of pool-completing
        # rows. Downstream no-op safety: scatter_paged_cache treats invalid
        # (-1) state slots as writes-to-nothing and the kpool compress kernel
        # masks flat_slot < 0, so the extra non-completing rows are harmless.
        # Eager keeps the min()/nonzero() forms.
        is_full_graph = (
            getattr(get_forward_context(), "cudagraph_runtime_mode", None)
            == CUDAGraphMode.FULL
        )
        num_tokens = (
            positions.shape[0]
            if is_full_graph
            else min(attn_metadata.num_actual_tokens, positions.shape[0])
        )
        k = k[:num_tokens].reshape(-1, self.head_dim)
        gate_score = gate_score[:num_tokens].reshape(-1, self.head_dim)
        current_state = torch.cat([k, gate_score], dim=-1).to(state_cache.dtype)
        state_slots = state_metadata.slot_mapping[:num_tokens]
        scatter_paged_cache(
            state_cache,
            state_slots,
            current_state,
            state_metadata.block_size,
        )

        # Tokens whose pool-granular indexer slot is valid complete a pool.
        selected = (
            torch.arange(num_tokens, device=k.device)
            if is_full_graph
            else (indexer_metadata.slot_mapping[:num_tokens] >= 0).nonzero().flatten()
        )
        if is_full_graph or selected.numel() > 0:
            token_ids = torch.arange(num_tokens, device=k.device)
            request_ids = torch.bucketize(
                token_ids,
                attn_metadata.cum_query_lens,
                right=True,
            ).clamp_max(attn_metadata.seq_lens.shape[0] - 1)
            pool_state = gather_compressor_state(
                state_cache,
                state_metadata.block_table,
                state_metadata.block_size,
                positions[selected],
                request_ids[selected],
                index_kpool,
            )
            # The permuted [blocks, block, 1, D] cache makes the gather return
            # [n, kpool, 1, D]; flatten the head dim for the pool math below.
            if pool_state.ndim == 4:
                pool_state = pool_state.reshape(pool_state.shape[0], pool_state.shape[1], -1)
            query_ends = attn_metadata.cum_query_lens
            query_offsets = torch.cat([torch.zeros_like(query_ends[:1]), query_ends[:-1]])
            query_lens = query_ends - query_offsets
            selected_request_ids = request_ids[selected]
            request_query_starts = attn_metadata.seq_lens[selected_request_ids] - query_lens[selected_request_ids]
            pool_offsets = torch.arange(
                index_kpool - 1,
                -1,
                -1,
                device=k.device,
            )
            pool_positions = positions[selected, None] - pool_offsets[None, :]
            local_positions = pool_positions - request_query_starts[:, None]
            current_mask = (local_positions >= 0) & (local_positions < query_lens[selected_request_ids, None])
            current_indices = (
                (query_offsets[selected_request_ids, None] + local_positions.clamp_min(0))
                .long()
                .clamp_max(num_tokens - 1)
            )
            current_pool_state = current_state[current_indices]
            pool_state = torch.where(
                current_mask.unsqueeze(-1),
                current_pool_state,
                pool_state,
            )
            pool_k, pool_gate = pool_state.split(self.head_dim, dim=-1)
            glm5_next_kpool_compress_and_write_cache(
                indexer_cache,
                pool_k.to(torch.bfloat16),
                pool_gate.to(torch.bfloat16),
                compress_ape,
                indexer_metadata.slot_mapping[selected].to(torch.int64),
            )

        # Top-k pool selection. The CANN pool_key_indexer wheel is not
        # available on CANN 9.1.0/910B, so selection runs in PyTorch: the
        # chunked eager scorer for prefill and a static-shape scorer for
        # FULL-graph decode capture (the eager path cannot be captured --
        # its loop bound comes from seq_lens.max().item() and its gathers
        # are data-length shaped).
        cum_query_lens = attn_metadata.cum_query_lens
        # Pool-level budget: expanding ``topk_tokens`` tokens from
        # ``index_kpool``-sized pools (history_group_budget_for_topk:
        # topk // pool_size). Selecting more pools over-attends relative to
        # the trained model.
        pool_budget = (
            self.topk_tokens // index_kpool
            if index_kpool > 1 and self.topk_tokens % index_kpool == 0
            else self.topk_tokens
        )
        if is_full_graph:
            indices = indexer_kpool_topk_static(
                q[:num_tokens],
                indexer_cache,
                weights[:num_tokens],
                cum_query_lens,
                indexer_metadata.seq_lens,
                indexer_metadata.block_table,
                positions[:num_tokens],
                sparse_count=pool_budget,
                pool_size=index_kpool,
            )
        else:
            indices = indexer_kpool_topk_pytorch(
                q[:num_tokens],
                indexer_cache,
                weights[:num_tokens],
                cum_query_lens,
                indexer_metadata.seq_lens,
                indexer_metadata.block_table,
                positions[:num_tokens],
                sparse_count=pool_budget,
                pool_size=index_kpool,
                max_key_seq_len=int(indexer_metadata.seq_lens.max().item()) if indexer_metadata.seq_lens.numel() else 0,
            )
        if self.topk_indices_buffer is not None:
            # The pytorch topk returns POOL ids; npu_sparse_flash_attention
            # consumes TOKEN indices (each selected pool contributes its
            # kpool constituent tokens). Expand — matching the NVIDIA
            # backend's ``expand_pools_to_tokens`` (logical token positions) —
            # append the per-query causal tail, then pad to the buffer width
            # ceil((topk + kpool - 1)/128)*128 with -1.
            valid = indices >= 0
            expanded = expand_pools_to_tokens(
                indices[:, :pool_budget],
                valid[:, :pool_budget],
                pool_budget * index_kpool if index_kpool > 1 else pool_budget,
                index_kpool,
            )
            expanded = self._append_causal_tail(
                expanded,
                positions[:num_tokens],
                indexer_metadata,
                cum_query_lens,
                index_kpool,
            )
            width = self.topk_indices_buffer.shape[1]
            if expanded.shape[1] > width:
                raise ValueError(
                    f"GLM-5 kpool topk width {expanded.shape[1]} exceeds the "
                    f"padded buffer width {width}."
                )
            if expanded.shape[1] < width:
                pad = torch.full(
                    (expanded.shape[0], width - expanded.shape[1]),
                    -1,
                    dtype=expanded.dtype,
                    device=expanded.device,
                )
                expanded = torch.cat([expanded, pad], dim=1)
            self.topk_indices_buffer[:num_tokens] = expanded
            return expanded.unsqueeze(1)
        return indices.unsqueeze(1)

    @staticmethod
    def _append_causal_tail(
        expanded: torch.Tensor,
        query_positions: torch.Tensor,
        indexer_metadata,
        cum_query_lens: torch.Tensor,
        index_kpool: int,
    ) -> torch.Tensor:
        """Append each query's un-pooled causal tail (up to kpool-1 tokens).

        The pipeline is select-pools -> expand -> append-tail: the most
        recent tokens that have not yet completed a pool are forced into
        the attended set regardless of indexer score, the current decode
        token included (the buffer width ceil((topk+kpool-1)/128)*128
        reserves exactly these kpool-1 slots).
        """
        if index_kpool <= 1 or expanded.shape[0] == 0:
            return expanded
        device = expanded.device
        # Per-query causal pool count, mirroring indexer_kpool_topk_pytorch:
        # pools fully inside [0, position] clamped to the cached pool count.
        token_ids = torch.arange(query_positions.shape[0], device=device)
        request_ids = torch.bucketize(
            token_ids, cum_query_lens, right=True
        ).clamp_max(indexer_metadata.seq_lens.shape[0] - 1)
        request_pool_lens = indexer_metadata.seq_lens[request_ids].to(torch.int64)
        causal_pools = torch.div(
            query_positions.to(torch.int64) + 1,
            index_kpool,
            rounding_mode="floor",
        )
        causal_pools = torch.minimum(causal_pools, request_pool_lens)
        tail_start = causal_pools * index_kpool
        tail_count = (query_positions.to(torch.int64) + 1) - tail_start
        offsets = torch.arange(index_kpool - 1, device=device)
        tail_raw = tail_start[:, None] + offsets[None, :]
        is_tail = offsets[None, :] < tail_count[:, None]
        tail = torch.where(
            is_tail,
            tail_raw,
            torch.full_like(tail_raw, -1),
        ).to(expanded.dtype)
        return torch.cat([expanded, tail], dim=1)
