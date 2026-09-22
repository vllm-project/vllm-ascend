# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense Ascend attention using flash-attention-npu's device-side FA3 tiling."""

from bisect import bisect_left
from dataclasses import dataclass, field

import torch
from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

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
    # spec -> tiling tensor consumed by each layer's forward. In graph mode this
    # aliases the builder's cache dictionary for the current graph_key; eager
    # execution owns a fresh dictionary. This field only references tiling storage.
    scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)


class AscendFlashAttentionMetadataBuilder(AttentionMetadataBuilder[AscendFlashAttentionMetadata]):
    # Per-request query offsets let one paged call handle mixed prefill/decode
    # and multi-token verification in the existing request order.
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
        self.capture_sizes = set(vllm_config.compilation_config.cudagraph_capture_sizes or [])
        self.block_size = kv_cache_spec.block_size
        # Deduplicate layer parameters so identical layers share one tiling call.
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
        # Two-level cache: scheduler_buffers[graph_key][spec] = tiling_tensor.
        # Outer key: (input-buffer addresses, captured token bound), identifying
        # the graph context. Its value is a dictionary, not a tiling tensor.
        # Inner key: the layer's operator spec (heads, head size, dtype, scale,
        # softcap). Its value is the device tiling tensor shared by matching layers.
        # scheduler_metadata aliases the selected inner dictionary, so writing
        # scheduler_metadata[spec] updates scheduler_buffers[graph_key][spec].
        # Retain these tensors across builds and copy new tiling into them to
        # preserve the device addresses recorded by graph capture.
        self.scheduler_buffers: dict[tuple, dict[tuple, torch.Tensor]] = {}

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
        # The runner can append zero-length or dummy requests for graph padding.
        # Its CPU offsets identify the active prefix without a device sync.
        # Keep one empty request for dummy forwards: the operator requires B > 0.
        active_reqs = max(
            1,
            bisect_left(common.query_start_loc_cpu.numpy(), common.num_actual_tokens, 0, num_reqs + 1),
        )
        # The runner owns graph-stable buffers. Device tiling limits reads to
        # active_reqs, so padding rows need neither copying nor sanitizing.
        query_start_loc = common.query_start_loc
        seq_lens = common.seq_lens
        block_tables = common.block_table_tensor
        num_input_tokens = common.num_input_tokens or common.num_actual_tokens
        is_graph_capture_size = num_input_tokens in self.capture_sizes
        max_query_len = num_input_tokens if is_graph_capture_size else common.max_query_len
        if is_graph_capture_size:
            # Distinct input buffers (including draft steps) and token buckets
            # must retain independent tiling storage even for identical specs.
            source_key = (query_start_loc.data_ptr(), seq_lens.data_ptr(), block_tables.data_ptr())
            graph_key = (source_key, max_query_len)
            # Select the inner spec -> tensor dictionary. The assignment below
            # to scheduler_metadata[spec] also updates scheduler_buffers[graph_key].
            scheduler_metadata = self.scheduler_buffers.setdefault(graph_key, {})
        else:
            scheduler_metadata = {}
        for spec in self.scheduler_specs:
            # Use the same layer configuration created in __init__ and used by
            # forward. Page size and causality are fixed for this attention group;
            # common.causal is still passed to tiling and attention, not hardcoded.
            num_heads, num_kv_heads, head_size, dtype, scale, softcap = spec
            # AICPU tiling runs once per layout, before entering the model graph.
            # The operator's auxiliary-stream events cannot themselves be
            # captured on all CANN releases. This still consumes device lengths
            # asynchronously and requires no host-side per-layer graph updates.
            tiling = get_scheduler_metadata(
                # Inactive slots must not participate in FlashDecode's
                # min-Q/task count, even when runner buffers include padding.
                batch_size=active_reqs,
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
                num_splits=0,
            )
            if spec in scheduler_metadata:
                scheduler_metadata[spec].copy_(tiling)
            else:
                scheduler_metadata[spec] = tiling
        return AscendFlashAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_tables=block_tables,
            slot_mapping=common.slot_mapping[: common.num_actual_tokens],
            max_query_len=max_query_len,
            causal=common.causal,
            model_runner_type=self.model_runner_type,
            scheduler_metadata=scheduler_metadata,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ) -> AscendFlashAttentionMetadata:
        # Keep the runner's capture signature; FA3 handles all scheduling states
        # through the same paged interface and does not consume attn_state.
        return self.build(0, common_attn_metadata)

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
        self.logits_soft_cap = logits_soft_cap or 0.0

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        # The metadata builder updates FA3 inputs and device tiling in place
        # before replay; no host-side attention tasks need to be updated.
        pass

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
            num_splits=0,
        )
        output.copy_(result)
        return output
