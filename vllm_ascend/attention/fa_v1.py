# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Device-length CANN FlashAttn for A5 V2 drafts."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask


@dataclass
class AscendFlashAttentionMetadata(AscendMetadata):
    flash_attn_mask: torch.Tensor | None = None
    # Share plans only within an attention group and draft step.
    flash_attn_plans: dict = field(default_factory=dict)
    flash_attn_lengths: tuple[torch.Tensor, torch.Tensor] | None = None
    flash_attn_executor: DeviceMetadataExecutor | None = None
    flash_attn_group_id: int = 0


@contextmanager
def prepare_flash_attn_metadata(attn_metadata, layers, executor, num_tokens):
    """Overlap draft tiling with projections, capturing both producer and waits."""
    if executor is None or attn_metadata is None:
        yield
        return
    if executor.submission_in_flight:
        # FULL draft capture starts before context KV; _run_model joins that scope.
        if any(attn_metadata[name].flash_attn_executor is not executor for name in layers):
            raise RuntimeError("Draft metadata does not match the active producer scope")
        yield
        return
    groups: dict[int, tuple[AscendFlashAttentionMetadata, list[AscendFlashAttentionImpl]]] = {}
    for group_id, (name, impl) in enumerate(layers.items()):
        metadata = attn_metadata[name]
        if id(metadata) not in groups:
            metadata.flash_attn_lengths = None
            metadata.flash_attn_plans.clear()
            metadata.flash_attn_executor = executor
            # Use the stable layer index, not the per-step metadata object's id.
            metadata.flash_attn_group_id = group_id
            groups[id(metadata)] = (metadata, [])
        groups[id(metadata)][1].append(impl)

    def prepare(metadata, implementations):
        for impl in implementations:
            impl.prepare_metadata(metadata, num_tokens)

    tasks = tuple(
        DeviceMetadataTask(
            DeviceMetadataStage.ATTENTION,
            lambda metadata=metadata, implementations=implementations: prepare(metadata, implementations),
            metadata.flash_attn_group_id,
        )
        for metadata, implementations in groups.values()
    )
    submitted = False
    try:
        executor.submit(tasks)
        submitted = True
        yield
    finally:
        stream = torch.npu.current_stream()
        if executor.submission_in_flight:
            if submitted:
                for task in tasks:
                    executor.wait(task.stage, task.group_id)
            else:
                # A producer failed before recording its readiness event.
                stream.wait_stream(executor.stream)
            executor.release()
        for metadata, _ in groups.values():
            # Aux-stream allocations must outlive the queued main-stream reads.
            tensors = list(metadata.flash_attn_lengths or ())
            tensors.extend(plan for plan, _ in metadata.flash_attn_plans.values())
            for tensor in tensors:
                tensor.record_stream(stream)
            metadata.flash_attn_executor = None


class AscendFlashAttentionMetadataBuilder(AscendAttentionMetadataBuilder):
    metadata_cls = AscendFlashAttentionMetadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build from exact device lengths; CPU mirrors may be stale after rejection."""
        num_tokens = common_attn_metadata.num_actual_tokens
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        return self.metadata_cls(
            num_actual_tokens=num_tokens,
            num_decode_tokens=num_tokens,
            num_decodes=num_reqs,
            block_tables=common_attn_metadata.block_table_tensor[:num_reqs],
            query_start_loc=query_start_loc,
            query_start_loc_gpu=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_gpu=seq_lens,
            max_query_len=common_attn_metadata.max_query_len,
            slot_mapping=common_attn_metadata.slot_mapping[:num_tokens],
            causal=common_attn_metadata.causal,
            attn_state=common_attn_metadata.attn_state,
            model_runner_type=self.model_config.runner_type,
            flash_attn_mask=self.attn_mask_builder.get_splitfuse_attn_mask(),
        )


class AscendFlashAttentionBackend(AscendAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return AscendFlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        return AscendFlashAttentionMetadataBuilder


class AscendFlashAttentionImpl(AscendAttentionBackendImpl):
    def prepare_metadata(self, metadata, num_tokens):
        """Build tiling without Q/K/V; share it across identical draft layers."""
        # Load CANN only in workers using this backend.
        from cann_ops_transformer.ops import flash_attn_metadata

        if metadata.flash_attn_lengths is None:
            num_reqs = metadata.block_tables.shape[0]
            boundaries = metadata.query_start_loc_gpu[: num_reqs + 1].int()
            # Dummy requests may have repeated or negative query boundaries.
            boundaries = boundaries.clamp(min=0, max=num_tokens).cummax(dim=0).values
            lengths = metadata.seq_lens_gpu[:num_reqs].int().clamp(min=1)
            metadata.flash_attn_lengths = boundaries, lengths

        layout_kv = "PA_BNBD" if self.use_bnsd_kv_cache else "PA_BBND"
        plan_key = (self.num_heads, self.num_kv_heads, self.head_size, self.sliding_window, layout_kv)
        cached = metadata.flash_attn_plans.get(plan_key)
        if cached is None:
            window = self.sliding_window - 1 if self.sliding_window is not None else -1
            # Both CANN operators require the same lengths, mask and layout.
            # Mask modes: 0 = unmasked, 3 = causal, 4 = sliding window.
            attrs = dict(
                cu_seqlens_q=metadata.flash_attn_lengths[0],
                seqused_kv=metadata.flash_attn_lengths[1],
                max_seqlen_q=metadata.max_query_len,
                max_seqlen_kv=-1,
                mask_mode=4 if self.sliding_window is not None else 3 if metadata.causal else 0,
                win_left=window,
                win_right=0 if metadata.causal and self.sliding_window is not None else window,
                layout_q="TND",
                layout_kv=layout_kv,
                layout_out="TND",
            )
            cached = flash_attn_metadata(self.num_heads, self.num_kv_heads, self.head_size, **attrs), attrs
            metadata.flash_attn_plans[plan_key] = cached
        return cached

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        from cann_ops_transformer.ops import flash_attn

        metadata = attn_metadata
        if metadata.flash_attn_executor is not None:
            metadata.flash_attn_executor.wait(DeviceMetadataStage.ATTENTION, metadata.flash_attn_group_id)
        cached = self.prepare_metadata(metadata, query.shape[0])
        plan, attrs = cached

        with attention_transfer_window():
            result, _ = flash_attn(
                query,
                self.key_cache,
                self.value_cache,
                block_table=metadata.block_tables,
                metadata=plan,
                attn_mask=metadata.flash_attn_mask if attrs["mask_mode"] else None,
                sinks=self.sinks,
                softmax_scale=self.scale,
                return_softmax_lse=False,
                **attrs,
            )
            # CANN returns a new tensor; vLLM requires its supplied output buffer.
            output[: query.shape[0]].copy_(result)
            return output
