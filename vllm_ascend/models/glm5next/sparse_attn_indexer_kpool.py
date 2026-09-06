# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer layer for the GLM-5.3-Flash kpool indexer.

GLM-5.3-Flash enables its sparse indexer only when the checkpoint sets
``index_topk``; with ``index_topk`` unset the model runs as dense NoPE MLA plus
KDA, which is the configuration vLLM Ascend currently supports.

The kpool forward chain follows ``glm-next-0806`` and runs entirely in torch /
triton, with no CUDA-only fallback: K and gate stay FP32 until a completed pool
is compressed into the BF16 indexer cache by
``glm5_next_kpool_state_compress_and_write_cache``, then
``glm5_next_lightning_indexer`` scores every visible pool, selects ``topk``
pool ids, expands them to tokens, and appends the always-preserved causal tail.
The resulting ``[T, 1, index_topk + index_kpool - 1]`` indices are written back
to the model-global ``topk_indices_buffer`` so the sparse MLA attention layer
can consume them, mirroring the upstream vLLM sparse-indexer contract.
"""

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse attention indexer op for the GLM-5.3-Flash kpool indexer.

    ``forward_oot`` is the Ascend path (the op is enabled on the NPU platform
    and dispatched through the out-of-tree branch); ``forward_native`` shares
    the same implementation so the chain is exercised on CPU by unit tests.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str | None,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        *,
        state_cache,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.state_cache = state_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache

    def update_cache(
        self,
        k: torch.Tensor,
        gate_score: torch.Tensor,
        compress_ape: torch.Tensor,
        positions: torch.Tensor,
        cum_query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Write FP32 state and completed BF16 pools before index selection.

        K must already be normalized and rope-applied in FP32. Query ends
        exclude the initial zero, and sequence lengths count original tokens.
        The scoring implementation supplies the same batch boundaries it uses
        for attention. Padded rows must have slot_mapping == -1.
        """
        from vllm_ascend.ops.glm5_next_kpool_state_compress import (
            glm5_next_kpool_state_compress_and_write_cache,
        )

        metadata = get_forward_context().attn_metadata
        state_metadata = metadata[self.state_cache.prefix]
        indexer_metadata = metadata[self.k_cache.prefix]
        state = self._bound_cache(self.state_cache)
        indexer = self._bound_cache(self.k_cache)
        num_tokens = k.shape[0]
        glm5_next_kpool_state_compress_and_write_cache(
            state,
            indexer,
            k,
            gate_score,
            compress_ape,
            positions,
            cum_query_lens,
            seq_lens,
            state_metadata.slot_mapping[:num_tokens],
            state_metadata.block_table,
            indexer_metadata.slot_mapping[:num_tokens],
            index_kpool=self.k_cache.compress_ratio,
        )

    @staticmethod
    def _bound_cache(layer) -> torch.Tensor:
        """Select the cache bound to the active virtual engine."""
        context = get_forward_context()
        virtual_engine = getattr(context, "virtual_engine", 0) or 0
        cache = layer.kv_cache
        if isinstance(cache, (list, tuple)):
            cache = cache[virtual_engine]
        if isinstance(cache, (list, tuple)) and len(cache) == 1:
            cache = cache[0]
        if not isinstance(cache, torch.Tensor):
            raise TypeError(
                f"GLM-5 Indexer cache {type(layer).__name__} is not bound."
            )
        return cache

    def _run_indexer_chain(
        self,
        q_values: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        gate_score: torch.Tensor,
        compress_ape: torch.Tensor,
        positions: torch.Tensor,
        index_kpool: int,
    ) -> torch.Tensor:
        """Write compressor state, select pool top-k, and expand to tokens."""
        if index_kpool <= 0:
            raise ValueError(f"index_kpool must be positive, got {index_kpool}.")

        from vllm_ascend.ops.glm5_next_lightning_indexer import (
            glm5_next_lightning_indexer,
        )

        context = get_forward_context()
        metadata = context.attn_metadata
        if not isinstance(metadata, dict):
            raise TypeError("GLM-5 Indexer requires per-layer attention metadata.")
        indexer_metadata = metadata[self.k_cache.prefix]
        state = self._bound_cache(self.state_cache)
        indexer = self._bound_cache(self.k_cache)
        if not isinstance(state, torch.Tensor):
            raise TypeError("GLM-5 compressor state cache must be one tensor.")
        if not isinstance(indexer, torch.Tensor) or indexer.dtype != torch.bfloat16:
            raise TypeError("GLM-5 indexer cache must be one bfloat16 K tensor.")
        if (
            indexer_metadata.cum_query_lens is None
            or indexer_metadata.raw_seq_lens is None
        ):
            raise ValueError("GLM-5 indexer metadata is missing request boundaries.")

        is_full_graph = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        # Eager MTP keeps the first-pass buffer length for later draft steps,
        # while the per-step attention metadata contains only the real query
        # rows. Do not feed those padded rows into cache/indexer addressing.
        # Full graphs must retain their captured fixed shape instead.
        num_tokens = (
            positions.shape[0]
            if is_full_graph
            else min(indexer_metadata.num_actual_tokens, positions.shape[0])
        )
        if num_tokens == 0:
            return torch.empty(
                (0, 1, self.topk_tokens + index_kpool - 1),
                dtype=torch.int32,
                device=q_values.device,
            )

        self.update_cache(
            k[:num_tokens],
            gate_score[:num_tokens],
            compress_ape,
            positions[:num_tokens],
            indexer_metadata.cum_query_lens,
            indexer_metadata.raw_seq_lens,
        )

        max_pool_seq_len = (
            indexer_metadata.block_table.shape[1] * indexer.shape[1]
            if is_full_graph
            else int(indexer_metadata.seq_lens_cpu.max())
        )
        topk = glm5_next_lightning_indexer(
            q_values[:num_tokens],
            indexer,
            weights[:num_tokens].to(q_values.dtype),
            indexer_metadata.cum_query_lens,
            indexer_metadata.seq_lens,
            indexer_metadata.block_table,
            positions[:num_tokens],
            index_topk=self.topk_tokens,
            index_kpool=index_kpool,
            max_pool_seq_len=max_pool_seq_len,
        )

        # The model-global buffer is padded out to a CANN-friendly multiple;
        # keep every padded column at -1 so a later sparse MLA reader that
        # slices the full buffer width never sees garbage.
        buffer_width = self.topk_tokens + index_kpool - 1
        self.topk_indices_buffer[:num_tokens].fill_(-1)
        self.topk_indices_buffer[:num_tokens, :buffer_width].copy_(topk.squeeze(1))
        return topk

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None,
        compress_ape: torch.Tensor | None,
        index_kpool: int,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        del hidden_states
        if self.use_fp4_cache:
            raise ValueError("Ascend GLM-5 Indexer uses BF16 Q, not FP4.")
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
            if q_scale is not None:
                q_values = q_values * q_scale.unsqueeze(-1).to(q_values.dtype)
        else:
            q_values = q_quant
        if gate_score is None or compress_ape is None or positions is None:
            raise ValueError(
                "GLM-5 kpool requires gate_score, compress_ape, and positions."
            )
        return self._run_indexer_chain(
            q_values,
            k,
            weights,
            gate_score,
            compress_ape,
            positions,
            index_kpool,
        )

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )
