# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense Ascend attention using flash-attention-npu's device-side FA3 tiling."""

from dataclasses import dataclass, field

import torch
from flash_attn_npu_3 import flash_attn_varlen_func, flash_attn_with_kvcache, get_scheduler_metadata
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder, AttentionType
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start


class AscendFlashAttentionBackend(AscendAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type["AscendFlashAttentionImpl"]:
        return AscendFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["AscendFlashAttentionMetadataBuilder"]:
        return AscendFlashAttentionMetadataBuilder


@dataclass
class AscendFlashAttentionMetadata(AscendMetadata):
    # Shared only within this execution, keyed by the operator's static parameters.
    scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)
    varlen_scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)


class AscendFlashAttentionMetadataBuilder(AttentionMetadataBuilder[AscendFlashAttentionMetadata]):
    # A single TND invocation handles mixed query lengths, including verification
    # of multiple speculative tokens. No decode-first batch reordering is needed.
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.device = device
        self.model_runner_type = vllm_config.model_config.runner_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs + 1
        self.capture_sizes = set(vllm_config.compilation_config.cudagraph_capture_sizes or [])
        self.block_size = kv_cache_spec.block_size
        self.scheduler_specs = set()
        for name in layer_names:
            impl = vllm_config.compilation_config.static_forward_context[name].impl
            self.scheduler_specs.add(
                (
                    impl.num_heads,
                    impl.num_kv_heads,
                    impl.head_size,
                    kv_cache_spec.dtype,
                    impl.scale,
                    impl.logits_soft_cap,
                )
            )
        self.scheduler_buffers: dict[tuple, torch.Tensor] = {}
        # FULL graphs can replay a different number of requests at the same
        # token count. Keep the operator's request dimensions and addresses fixed.
        # Draft steps have distinct runner-owned input buffers and must not alias.
        self.graph_buffers: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendFlashAttentionMetadata:
        common = common_attn_metadata
        num_reqs = common.num_reqs
        if num_reqs > self.max_num_reqs:
            raise ValueError("FA3 request count exceeds the configured batch capacity.")
        source_key = (
            common.query_start_loc.data_ptr(),
            common.seq_lens.data_ptr(),
            common.block_table_tensor.data_ptr(),
        )
        if source_key not in self.graph_buffers:
            self.graph_buffers[source_key] = (
                torch.empty(self.max_num_reqs + 1, dtype=torch.int32, device=self.device),
                torch.empty(self.max_num_reqs, dtype=torch.int32, device=self.device),
                torch.empty(
                    (self.max_num_reqs, common.block_table_tensor.shape[1]), dtype=torch.int32, device=self.device
                ),
            )
        query_start_loc, seq_lens, block_tables = self.graph_buffers[source_key]
        # The runner may add a dummy FIA request beyond the seq_lens/block-table
        # capacity. FA3 needs no such request: zero query lengths skip padding.
        query_start_loc.fill_(common.num_actual_tokens)
        query_start_loc[: num_reqs + 1].copy_(common.query_start_loc[: num_reqs + 1])
        query_start_loc.clamp_(min=0, max=common.num_actual_tokens)
        seq_lens.zero_()
        seq_count = min(num_reqs, common.seq_lens.shape[0])
        seq_lens[:seq_count].copy_(common.seq_lens[:seq_count])
        block_tables.zero_()
        block_count = min(num_reqs, common.block_table_tensor.shape[0])
        block_tables[:block_count].copy_(common.block_table_tensor[:block_count])
        num_input_tokens = common.num_input_tokens or common.num_actual_tokens
        graph_shape = num_input_tokens in self.capture_sizes
        max_query_len = num_input_tokens if graph_shape else common.max_query_len
        scheduler_metadata = {}
        varlen_scheduler_metadata = {}
        if common.attn_state == AscendAttentionState.PrefillNoCache:
            # Varlen consumes only the current packed K/V. Prepare its tiling
            # once per layer layout, including warmups for capturable shapes.
            varlen_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            for spec in self.scheduler_specs:
                num_heads, num_kv_heads, head_size, dtype, scale, softcap = spec
                varlen_scheduler_metadata[spec] = get_scheduler_metadata(
                    batch_size=self.max_num_reqs,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_query_len,
                    num_heads_q=num_heads,
                    num_heads_kv=num_kv_heads,
                    headdim=head_size,
                    cache_seqlens=varlen_kv_lens,
                    qkv_dtype=dtype,
                    cu_seqlens_q=query_start_loc,
                    page_size=None,
                    causal=common.causal,
                    softmax_scale=scale,
                    softcap=softcap,
                    num_splits=1,
                )
        needs_paged_attention = graph_shape or common.attn_state != AscendAttentionState.PrefillNoCache
        for spec in self.scheduler_specs if needs_paged_attention else ():
            num_heads, num_kv_heads, head_size, dtype, scale, softcap = spec
            # AICPU tiling runs once per layout, before entering the model graph.
            # The operator's auxiliary-stream events cannot themselves be
            # captured on all CANN releases. This still consumes device lengths
            # asynchronously and requires no host-side per-layer graph updates.
            tiling = get_scheduler_metadata(
                batch_size=self.max_num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=block_tables.shape[1] * self.block_size,
                num_heads_q=num_heads,
                num_heads_kv=num_kv_heads,
                headdim=head_size,
                cache_seqlens=seq_lens,
                qkv_dtype=dtype,
                cu_seqlens_q=query_start_loc,
                page_size=self.block_size,
                causal=common.causal,
                softmax_scale=scale,
                softcap=softcap,
                num_splits=1,
            )
            if graph_shape:
                buffer_key = (source_key, spec, max_query_len, common.causal)
                if buffer_key not in self.scheduler_buffers:
                    self.scheduler_buffers[buffer_key] = tiling
                else:
                    self.scheduler_buffers[buffer_key].copy_(tiling)
                tiling = self.scheduler_buffers[buffer_key]
            scheduler_metadata[spec] = tiling
        return AscendFlashAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_tables=block_tables,
            slot_mapping=common.slot_mapping[: common.num_actual_tokens],
            max_query_len=max_query_len,
            attn_state=common.attn_state,
            causal=common.causal,
            model_runner_type=self.model_runner_type,
            scheduler_metadata=scheduler_metadata,
            varlen_scheduler_metadata=varlen_scheduler_metadata,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ) -> AscendFlashAttentionMetadata:
        metadata = self.build(0, common_attn_metadata)
        metadata.attn_state = attn_state
        return metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: AscendCommonAttentionMetadata
    ) -> AscendFlashAttentionMetadata:
        return self.build_for_graph_capture(common_attn_metadata)


class AscendFlashAttentionImpl(AscendAttentionBackendImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
            **kwargs,
        )
        if self.enable_c8_quant or self.kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("FA3 requires FP16/BF16 KV cache; C8 and other quantized KV caches are unsupported.")
        if self.attn_type != AttentionType.DECODER:
            raise ValueError("FA3 currently supports decoder self-attention only.")
        if self.alibi_slopes is not None or self.sliding_window is not None or self.sinks is not None:
            raise ValueError("FA3 backend currently requires full attention without ALiBi or attention sinks.")
        parallel = self.vllm_config.parallel_config
        if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
            raise ValueError("FA3 context parallelism is not implemented.")
        self.logits_soft_cap = logits_soft_cap or 0.0

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendFlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        record_attention_compute_start()
        max_query_len = attn_metadata.max_query_len
        scheduler_key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            query.dtype,
            self.scale,
            self.logits_soft_cap,
        )
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache and not _EXTRA_CTX.capturing:
            num_tokens = attn_metadata.num_actual_tokens
            result = flash_attn_varlen_func(
                query[:num_tokens],
                key[:num_tokens],
                value[:num_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.varlen_scheduler_metadata[scheduler_key],
                num_splits=1,
            )
            output[:num_tokens].copy_(result)
            return output

        result = flash_attn_with_kvcache(
            query,
            self.key_cache,
            self.value_cache,
            cache_seqlens=attn_metadata.seq_lens,
            page_table=attn_metadata.block_tables,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=max_query_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata[scheduler_key],
            num_splits=1,
        )
        output.copy_(result)
        return output
