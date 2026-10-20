# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

try:
    from cann_ops_transformer.ops.ds41 import indexer_prologue_k
except ImportError:
    # A3 environments may not ship the ds41 operator package; the packed
    # cache path below requires it and raises a clear error when missing.
    indexer_prologue_k = None

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    scatter_cache_sk,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_v41.cache_config import (
    INDEX_FOLD_ROWS,
    make_folded_index_cache_spec,
    make_index_cache_spec,
    uses_a5_packed_cache,
)
from vllm_ascend.models.deepseek_v41.mixed_quant_attention import quantize_query_mxfp4
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz
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
        self.candidate_block_size = int(getattr(config, "candidate_block_size", INDEX_FOLD_ROWS))
        # FP32 copy of the K RMSNorm weight for ``indexer_prologue_k``; stays
        # ``None`` until ``process_weights_after_loading`` runs.
        self._prologue_norm_weight: torch.Tensor | None = None
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

    def process_weights_after_loading(self):
        """Prepare the fused indexer_prologue_k operands after weight loading."""
        if not self.owns_k:
            return
        if self.packed_cache_ops is not None:
            # The fused indexer_prologue_k requires its wk operand in Nz
            # format. This is a local requirement of the prologue op, so the
            # conversion is forced here instead of relying on the global
            # weight_nz_mode=2, which would also convert the MoE gate weight
            # that the A5 npu_fused_matmul gate path requires in ND format.
            if torch_npu.get_npu_format(self.wk.weight.data) != ACL_FORMAT_FRACTAL_NZ:
                self.wk.weight.data = torch_npu.npu_format_cast(self.wk.weight.data, ACL_FORMAT_FRACTAL_NZ)
        else:
            # Non-quantized caches keep the eager chain: follow the global policy.
            self.wk.weight.data = maybe_trans_nz(self.wk.weight.data)
        self._prologue_norm_weight = self.k_norm.weight.detach().to(torch.float32)

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        if not self.owns_k or latent.shape[0] == 0:
            return
        k_cache, scale_cache = self.k_cache.kv_cache[0]
        if self.packed_cache_ops is not None:
            if slots.ndim == 1:
                # Builder-prepared flat slots; identical to converting the
                # [T, 2] mapping per call, computed once per cache group.
                slot_mapping = slots
            elif slots.ndim == 2:
                slot_mapping = (slots[:, 0]) * k_cache.shape[1] + slots[:, 1]
                slot_mapping = slot_mapping.clamp(min=-1).to(torch.int32)
            else:
                raise ValueError(f"indexer update_keys slot mapping must be [T, 2] or [T], got ndim={slots.ndim}")
            cache_index = slot_mapping.to(torch.int64)
            norm_weight = self._prologue_norm_weight
            if norm_weight is None:
                norm_weight = self.k_norm.weight.float()
            prologue_common = dict(
                latent=latent,
                wk=self.wk.weight.detach(),
                norm_weight=norm_weight,
                rope_sin=sin.reshape(-1, self.rope_width),
                rope_cos=cos.reshape(-1, self.rope_width),
                norm_eps=self.k_norm.eps,
            )
            if indexer_prologue_k is None:
                raise RuntimeError(
                    "cann_ops_transformer.ops.ds41.indexer_prologue_k is required "
                    "on the A5 packed-cache path; install the ds41 operator wheels"
                )
            # storage_mode=0 writes the separated K/scale planes (QLI view);
            # the operator folds rows internally for storage_mode=1.
            indexer_prologue_k(
                k_cache=k_cache,
                k_scale_cache=scale_cache,
                cache_index=cache_index,
                storage_mode=0,
                combined_block_size=-1,
                **prologue_common,
            )
            if self.k_cache_folded is not None:
                if self.candidate_block_size != INDEX_FOLD_ROWS:
                    raise ValueError(
                        f"folded cache packs {INDEX_FOLD_ROWS} rows per row but "
                        f"candidate_block_size is {self.candidate_block_size}; the ds41 "
                        "prologue folds one candidate block per folded row"
                    )
                indexer_prologue_k(
                    k_cache=self.k_cache_folded.kv_cache[0],
                    k_scale_cache=None,
                    cache_index=cache_index,
                    storage_mode=1,
                    combined_block_size=self.candidate_block_size,
                    **prologue_common,
                )
            return
        key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
        DeviceOperator.apply_partial_rotary_inplace(key, cos, sin, start=self.width - self.rope_width, end=self.width)
        key = key.squeeze(1)
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
            return quantize_query_mxfp4(query)
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
