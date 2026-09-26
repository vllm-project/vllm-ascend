# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    scatter_cache_sk,
)
from vllm_ascend.models.deepseek_v41.cache_config import DeepseekV41IndexerSpec
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)

from .compressor import DeepseekV41RMSNorm


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
    ):
        super().__init__()
        self.owns_k = owns_k
        self.compress_ratio = compress_ratio
        self.n_heads = int(config.index_n_heads)
        self.width = int(config.index_head_dim)
        self.rope_width = int(config.qk_rope_head_dim)
        self.index_topk = int(config.index_topk)
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
            self.k_norm = DeepseekV41RMSNorm(self.width, config.rms_norm_eps)
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                DeepseekV41IndexerSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.width,
                    dtype=torch.int8,
                    tokens_per_state=compress_ratio,
                    scale_dim=1,
                    scale_dtype=torch.float16,
                ),
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
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            key.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        key = key.squeeze(1)
        quantized, scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
        k_cache, scale_cache = self.k_cache.kv_cache[0]
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
        output_indices=None,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        query = self._output(self.wq_b, qr).unflatten(-1, (self.n_heads, self.width))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        weights = self._output(self.weights_proj, hidden_states)
        weights = weights.float() * self.weights_scale

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
            output_indices=output_indices,
        )

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
        output_indices=None,
    ):
        """Run QLI V2 on paged INT8 K; candidates are block IDs, not positions.

        Source and consumer share [tokens, 1, candidate_topk_blocks] INT32
        block IDs only within this forward. Query quantization and position
        ordering stay outside the native QLI/candidate operator.
        """
        if is_candidate_source and uses_candidate_filter:
            raise ValueError("A candidate source must use the unfiltered position TopK")
        if uses_candidate_filter and candidates is None:
            raise RuntimeError("V4.1 candidate-filtering indexer ran before its source")
        if self.width != 128 or self.n_heads not in (32, 64):
            raise ValueError("QLI requires index_head_dim=128 and 32 or 64 index heads")
        if not 1 <= self.index_topk <= 2048:
            raise ValueError("QLI requires index_topk in [1, 2048]")
        if self.compress_ratio not in (1, 2):
            raise ValueError("Aurora QLI supports compression ratios 1 and 2")
        if is_candidate_source or uses_candidate_filter:
            if not 0 < candidate_topk_blocks <= 2048 or candidate_topk_blocks % 64:
                raise ValueError("candidate_topk_blocks must be a multiple of 64 in [64, 2048]")
            if candidate_block_size != 8:
                raise ValueError("The candidate kernel requires candidate_block_size=8")
        candidate_shape = (query.shape[0], 1, candidate_topk_blocks)
        if uses_candidate_filter and (candidates.shape != candidate_shape or candidates.dtype != torch.int32):
            raise ValueError("Candidate consumer requires INT32 block IDs with matching query rows")
        topk = self.index_topk
        if query.shape[0] == 0:
            selected = torch.full((0, topk), -1, dtype=torch.int32, device=query.device)
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates

        quantized_query, query_scale = quantize_indexer_query(query)
        weights = weights.to(torch.float16)
        key, key_scale = source_cache
        key_scale = key_scale.squeeze(-1)  # Preserve the Hybrid cache page stride.
        cu_seqlens_q = source_metadata.query_start_loc
        seqused_k = source_metadata.cache_seq_lens
        residual = source_metadata.cmp_residual
        if self.compress_ratio != 1 and residual is None:
            seq = getattr(source_metadata, "seq_lens", None)
            n = int(getattr(source_metadata, "num_reqs", 0) or (seq.shape[0] if seq is not None else 0))
            if seq is not None and n:
                residual = torch.remainder(seq[:n], self.compress_ratio).to(dtype=torch.int32)
            else:
                residual = torch.zeros((max(n, 1),), dtype=torch.int32, device=query.device)
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
        if op_metadata is None:
            raise RuntimeError("V4.1 QLI metadata was not built")
        if not getattr(source_metadata, "_skip_qli_wait", False):
            wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
        # DAV_3510 / A5 QLI has no two-level candidate TopK (source=1, consumer=2).
        # Main v2 is 2-output and drops candidate_out; v3 matches DSV4F's v2.
        mode = 3
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
            candidate_topk_index=None,
            candidate_mode=mode,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **common,
        )
        selected = prepare_indexer_indices(
            selected.squeeze(1),
            positions,
            self.compress_ratio,
            output=output_indices,
        )
        return selected, candidate_out if is_candidate_source else candidates
