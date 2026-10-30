# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    scatter_cache_sk,
)
from vllm_ascend.attention.dsa_v41_indexer_shard import (
    AscendDSAV41PrefillRowShard,
    gather_prefill_row_shard_rows,
    pack_prefill_row_shard_rows,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_v41.cache_config import (
    make_folded_index_cache_spec,
    make_index_cache_spec,
    uses_a5_packed_cache,
)
from vllm_ascend.ops.triton.fold_indexer_cache import fold_indexer_cache_rows
from vllm_ascend.ops.triton.prepare_indexer_indices import (
    prepare_indexer_indices,
    register_indexer_warmup,
)
from vllm_ascend.ops.triton.quantize_indexer_query import (
    _QUANTIZE_INDEXER_QUERY_KERNEL,
    quantize_indexer_query,
)
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import quantize_mxfp4_indexer
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)


class DeepseekV41Indexer(nn.Module):
    """Small side attention that selects compressed KV positions.

    All index heads are replicated on each TP rank for the correctness path,
    so every rank produces identical sparse indices without an all-reduce.
    """

    def __init__(
        self,
        config,
        owns_k,
        vllm_config,
        prefix,
        compress_ratio,
        quant_config=None,
        is_candidate_source=False,
    ):
        super().__init__()
        self.owns_k = owns_k
        self.packed_cache_ops = DeviceOperator.get_dsv41_packed_cache_ops()
        self.compress_ratio = compress_ratio
        self.n_heads = int(config.index_n_heads)
        self.width = int(config.index_head_dim)
        self.rope_width = int(config.qk_rope_head_dim)
        self.index_topk = int(config.index_topk)
        # Declarative JIT warmup for both indexer kernels; the model runner
        # holds the registry window open during model construction.
        register_indexer_warmup(self.index_topk, compress_ratio, vllm_config)
        _QUANTIZE_INDEXER_QUERY_KERNEL.register_warmup(vllm_config)
        self.softmax_scale = self.width**-0.5
        self.weights_scale = self.softmax_scale * self.n_heads**-0.5
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        if owns_k:
            self.wk = nn.Linear(
                config.head_dim,
                self.width,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = RMSNorm(self.width, eps=config.rms_norm_eps, dtype=torch.bfloat16)
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                make_index_cache_spec(
                    block_size=vllm_config.cache_config.block_size,
                    head_size=self.width,
                    compress_ratio=compress_ratio,
                ),
            )
            self.k_cache_folded = (
                DeepseekV41CacheLayer(
                    vllm_config,
                    f"{prefix}.k_cache_folded",
                    make_folded_index_cache_spec(block_size=vllm_config.cache_config.block_size),
                )
                if is_candidate_source and uses_a5_packed_cache()
                else None
            )

    @staticmethod
    def _output(linear, value):
        output = linear(value)
        return output[0] if isinstance(output, tuple) else output

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        if not self.owns_k or latent.shape[0] == 0:
            return
        key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
        DeviceOperator.apply_partial_rotary_inplace(key, cos, sin, start=self.width - self.rope_width, end=self.width)
        key = key.squeeze(1)
        k_cache, scale_cache = self.k_cache.kv_cache[0]
        if self.packed_cache_ops is not None:
            self.packed_cache_ops.write_index_cache((k_cache, scale_cache), slots, key)
            if self.k_cache_folded is not None:
                fold_indexer_cache_rows((k_cache, scale_cache), self.k_cache_folded.kv_cache[0], slots)
            return
        quantized, scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
        scatter_cache_sk(k_cache, slots, quantized)
        scatter_cache_sk(
            scale_cache,
            slots,
            scale.unsqueeze(-1).to(torch.float16),
        )

    def select(
        self,
        hidden_states,
        qr,
        positions,
        cos,
        sin,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
        candidate_lengths=None,
        topk_lengths=None,
        indices_output=None,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        query = self.project_query(qr)
        self.apply_query_rope(query, cos, sin)
        weights = self.project_weights(hidden_states)

        return self.select_projected(
            query,
            weights,
            positions,
            source_cache,
            source_metadata,
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            candidates=candidates,
            candidate_lengths=candidate_lengths,
            topk_lengths=topk_lengths,
            indices_output=indices_output,
        )

    def project_query(self, qr):
        return self._output(self.wq_b, qr).unflatten(-1, (self.n_heads, self.width))

    def apply_query_rope(self, query, cos, sin):
        DeviceOperator.apply_partial_rotary_inplace(query, cos, sin, start=self.width - self.rope_width, end=self.width)

    def project_weights(self, hidden_states):
        return self._output(self.weights_proj, hidden_states).float() * self.weights_scale

    def quantize_query(self, query):
        if self.packed_cache_ops is not None:
            return quantize_mxfp4_indexer(query)
        return quantize_indexer_query(query)

    def select_projected(
        self,
        query,
        weights,
        positions,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
        candidate_lengths=None,
        topk_lengths=None,
        indices_output=None,
        quantized_query=None,
        query_scale=None,
    ):
        """Run QLI V2 on paged INT8 K; candidates are block IDs, not positions.

        Source and consumer share [tokens, 1, candidate_topk_blocks] INT32
        block IDs only within this forward. Query quantization and position
        ordering stay outside the native QLI/candidate operator.
        """
        candidate_shape = (query.shape[0], 1, candidate_topk_blocks)
        topk = self.index_topk
        if query.shape[0] == 0:
            selected = (
                torch.full((0, topk), -1, dtype=torch.int32, device=query.device)
                if indices_output is None
                else indices_output
            )
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates
        if source_metadata.max_cache_seq_len == 0:
            if indices_output is not None:
                indices_output.fill_(-1)
            if topk_lengths is not None:
                topk_lengths.zero_()
            if is_candidate_source and candidate_lengths is not None:
                candidate_lengths.zero_()
            selected = (
                torch.full((query.shape[0], 0), -1, dtype=torch.int32, device=query.device)
                if indices_output is None
                else indices_output
            )
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates

        if self.packed_cache_ops is not None:
            return self.packed_cache_ops.run_a5_indexer(
                query,
                weights,
                positions,
                source_cache,
                source_metadata,
                topk=topk,
                compress_ratio=self.compress_ratio,
                is_candidate_source=is_candidate_source,
                uses_candidate_filter=uses_candidate_filter,
                candidate_topk_blocks=candidate_topk_blocks,
                candidate_block_size=candidate_block_size,
                candidates=candidates,
                candidate_lengths=candidate_lengths,
                topk_lengths=topk_lengths,
                indices_output=indices_output,
                quantized_query=quantized_query,
                query_scale=query_scale,
            )

        # TP prefill row shard: each rank scores only its contiguous slice of
        # prefill rows and the group exchanges the final int32 top-k. The
        # candidate source layer stays replicated (its published candidate
        # blocks must be complete on every rank for the consumer indexers).
        shard = getattr(source_metadata, "prefill_row_shard", None)
        if shard is not None and not is_candidate_source:
            return self._select_projected_sharded(
                shard=shard,
                query=query,
                weights=weights,
                positions=positions,
                source_cache=source_cache,
                quantized_query=quantized_query,
                query_scale=query_scale,
                is_candidate_source=is_candidate_source,
                uses_candidate_filter=uses_candidate_filter,
                candidate_topk_blocks=candidate_topk_blocks,
                candidate_block_size=candidate_block_size,
                candidates=candidates,
                topk_lengths=topk_lengths,
                indices_output=indices_output,
                topk=topk,
            )

        if quantized_query is None:
            quantized_query, query_scale = self.quantize_query(query)
        weights = weights.to(torch.float16)
        key, key_scale = source_cache
        key_scale = key_scale.squeeze(-1)  # Preserve the Hybrid cache page stride.
        cu_seqlens_q = source_metadata.query_start_loc
        seqused_k = source_metadata.cache_seq_lens
        residual = source_metadata.cmp_residual
        common = dict(
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            cmp_residual_k=residual,
            max_seqlen_q=source_metadata.max_query_len,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
        )
        op_metadata = source_metadata.qli_metadata
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
        mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
        selected, _, candidate_out = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            quantized_query,
            key,
            weights,
            query_scale,
            key_scale,
            topk,
            2,
            block_table=source_metadata.block_table,
            metadata=op_metadata,
            candidate_topk_index=candidates if uses_candidate_filter else None,
            candidate_mode=mode,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **common,
        )
        selected = prepare_indexer_indices(
            selected.squeeze(1),
            positions,
            self.compress_ratio,
            indices_output=indices_output,
            lengths_output=topk_lengths,
        )
        return selected, candidate_out if is_candidate_source else candidates

    def _select_projected_sharded(
        self,
        *,
        shard: AscendDSAV41PrefillRowShard,
        query: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        source_cache: tuple[torch.Tensor, ...],
        quantized_query: torch.Tensor | None,
        query_scale: torch.Tensor | None,
        is_candidate_source: bool,
        uses_candidate_filter: bool,
        candidate_topk_blocks: int,
        candidate_block_size: int,
        candidates: torch.Tensor | None,
        topk_lengths: torch.Tensor | None,
        indices_output: torch.Tensor | None,
        topk: int,
    ):
        """Score only this rank's prefill rows, then gather the full top-k.

        Each row still scores its FULL visible key history (the rebased
        request keeps the window's right end), so the gathered selection is
        bit-identical to the replicated path's.
        """
        query = pack_prefill_row_shard_rows(query, shard)
        weights = pack_prefill_row_shard_rows(weights, shard).to(torch.float16)
        positions = pack_prefill_row_shard_rows(positions, shard)
        if quantized_query is None:
            quantized_query, query_scale = self.quantize_query(query)
        else:
            quantized_query = pack_prefill_row_shard_rows(quantized_query, shard)
            query_scale = pack_prefill_row_shard_rows(query_scale, shard)
        packed_candidates = (
            pack_prefill_row_shard_rows(candidates, shard) if uses_candidate_filter else None
        )
        key, key_scale = source_cache
        key_scale = key_scale.squeeze(-1)  # Preserve the Hybrid cache page stride.

        # Rank-local QLI: feed the shard's rebased request geometry. Mode 1
        # never reaches here (the candidate source declines the shard).
        mode = 2 if uses_candidate_filter else 3
        op_metadata = shard.qli_metadata
        if op_metadata is not None:
            wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
        selected, _, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            quantized_query,
            key,
            weights,
            query_scale,
            key_scale,
            topk,
            2,
            block_table=shard.block_table,
            metadata=op_metadata,
            candidate_topk_index=packed_candidates,
            candidate_mode=mode,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            cu_seqlens_q=shard.qli_cu_seqlens_q,
            seqused_k=shard.qli_seqused_k,
            cmp_residual_k=shard.qli_cmp_residual_k,
            max_seqlen_q=shard.max_query_len,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
        )
        # post-process (filter/sort/lengths) is per-row, so run it on the
        # local rows before the exchange.
        local_lengths = (
            torch.empty((selected.shape[0], 1), dtype=torch.int32, device=selected.device)
            if topk_lengths is not None
            else None
        )
        selected = prepare_indexer_indices(
            selected.squeeze(1),
            positions,
            self.compress_ratio,
            lengths_output=local_lengths,
        )
        leading = selected[: shard.num_decode_tokens] if shard.num_decode_tokens else None
        logger.debug(
            "DSV41 indexer TP shard: rank scoring rows [%d, %d) of %d "
            "(local=%d, sizes=%s, topk=%d, ratio=%d) then all-gather",
            shard.start,
            shard.stop,
            shard.num_decode_tokens + sum(shard.sizes),
            shard.stop - shard.start,
            shard.sizes,
            topk,
            self.compress_ratio,
        )
        gathered = gather_prefill_row_shard_rows(
            selected[shard.num_decode_tokens :], shard.sizes, leading
        )
        if indices_output is not None:
            indices_output.copy_(gathered)
            gathered = indices_output
        if local_lengths is not None and topk_lengths is not None:
            leading_lengths = local_lengths[: shard.num_decode_tokens] if shard.num_decode_tokens else None
            gathered_lengths = gather_prefill_row_shard_rows(
                local_lengths[shard.num_decode_tokens :], shard.sizes, leading_lengths
            )
            topk_lengths.copy_(gathered_lengths)
        # Candidate blocks stay replicated: no producer path reaches here.
        return gathered, candidates
