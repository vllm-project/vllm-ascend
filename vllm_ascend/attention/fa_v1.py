# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Device-length CANN FlashAttn for A5 V2 drafts."""

from dataclasses import dataclass, field, replace
from typing import cast

import torch
import torch.nn.functional as F
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.compilation.updatable_graph import get_capture_resource
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window


@dataclass
class AscendFlashAttentionMetadata(AscendMetadata):
    flash_attn_mask: torch.Tensor | None = None
    # Eager plans belong to one attention group and draft step.
    flash_attn_plans: dict = field(default_factory=dict)
    flash_attn_lengths: tuple[torch.Tensor, torch.Tensor] | None = None
    captured_flash_attn_plans: dict = field(default_factory=dict)


class AscendFlashAttentionMetadataBuilder(AscendAttentionMetadataBuilder):
    metadata_cls = AscendFlashAttentionMetadata
    supports_draft_decode_metadata_update = True

    def update_draft_decode_metadata(self, metadata: AscendFlashAttentionMetadata) -> None:
        # Warmup uses the eager path. Only attach plans produced inside capture.
        if not torch.npu.is_current_stream_capturing():
            return

        layers = get_layers_from_vllm_config(self.vllm_config, Attention, self.layer_names)
        num_tokens = metadata.num_actual_tokens
        lengths = AscendFlashAttentionImpl._build_lengths(metadata, num_tokens)
        plans = {}
        for layer in layers.values():
            impl = cast(AscendFlashAttentionImpl, layer.impl)
            head_size = AscendFlashAttentionBackend.get_padded_head_size(impl.head_size)
            plan_key = impl._plan_key(metadata, head_size)
            key = (num_tokens, plan_key)
            if key not in plans:
                plans[key] = impl._build_plan(plan_key, lengths)
        metadata.captured_flash_attn_plans = plans

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
    def get_supported_head_sizes():
        # Logical 192-wide heads use zero padding to the native 256-wide kernel.
        return [64, 72, 128, 192, 256]

    @staticmethod
    def get_padded_head_size(head_size):
        return 256 if head_size == 192 else head_size

    @classmethod
    def customize_spec(cls, spec):
        head_size = cls.get_padded_head_size(spec.head_size)
        head_size_v = cls.get_padded_head_size(spec.head_size_v)
        if (head_size, head_size_v) == (spec.head_size, spec.head_size_v):
            return spec
        padded = replace(spec, head_size=head_size, head_size_v=head_size_v, page_size_padded=None)
        if spec.page_size_padded is not None:
            padded = replace(padded, page_size_padded=max(spec.page_size_padded, padded.real_page_size_bytes))
        return padded

    @staticmethod
    def get_impl_cls():
        return AscendFlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        return AscendFlashAttentionMetadataBuilder


class AscendFlashAttentionImpl(AscendAttentionBackendImpl):
    @staticmethod
    def _pad_head(tensor):
        padded_size = AscendFlashAttentionBackend.get_padded_head_size(tensor.shape[-1])
        if padded_size == tensor.shape[-1]:
            return tensor
        return F.pad(tensor, (0, padded_size - tensor.shape[-1]))

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        # Pad only newly written tokens; the persistent cache is allocated wide.
        return super().reshape_and_cache(
            query, self._pad_head(key), self._pad_head(value), kv_cache, attn_metadata, output
        )

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        return super().do_kv_cache_update(layer, self._pad_head(key), self._pad_head(value), kv_cache, slot_mapping)

    @staticmethod
    def _build_lengths(metadata, num_tokens):
        num_reqs = metadata.block_tables.shape[0]
        boundaries = metadata.query_start_loc_gpu[: num_reqs + 1].int()
        # Dummy requests may have repeated or negative query boundaries.
        boundaries = boundaries.clamp(min=0, max=num_tokens).cummax(dim=0).values
        lengths = metadata.seq_lens_gpu[:num_reqs].int().clamp(min=1)
        return boundaries, lengths

    def _plan_key(self, metadata, head_size):
        return (
            self.num_heads,
            self.num_kv_heads,
            head_size,
            self.sliding_window,
            "PA_BNBD" if self.use_bnsd_kv_cache else "PA_BBND",
            metadata.max_query_len,
            metadata.causal,
        )

    @staticmethod
    def _build_plan(plan_key, lengths):
        from cann_ops_transformer.ops import flash_attn_metadata

        num_heads, num_kv_heads, head_size, sliding_window, layout_kv, max_query_len, causal = plan_key
        window = sliding_window - 1 if sliding_window is not None else -1
        # Both CANN operators require the same lengths, mask and layout.
        # Mask modes: 0 = unmasked, 3 = causal, 4 = sliding window.
        attrs = dict(
            cu_seqlens_q=lengths[0],
            seqused_kv=lengths[1],
            max_seqlen_q=max_query_len,
            max_seqlen_kv=-1,
            mask_mode=4 if sliding_window is not None else 3 if causal else 0,
            win_left=window,
            win_right=0 if causal and sliding_window is not None else window,
            layout_q="TND",
            layout_kv=layout_kv,
            layout_out="TND",
        )
        return flash_attn_metadata(num_heads, num_kv_heads, head_size, **attrs), attrs

    def _get_plan(self, metadata, num_tokens, head_size):
        plan_key = self._plan_key(metadata, head_size)
        if torch.npu.is_current_stream_capturing():
            captured_plan = metadata.captured_flash_attn_plans.get((num_tokens, plan_key))
            if captured_plan is not None:
                return captured_plan
            # Mixed-backend graphs can skip the builder hook altogether.
            lengths = get_capture_resource(
                ("flash_attn.lengths", id(metadata), num_tokens),
                lambda: self._build_lengths(metadata, num_tokens),
            )
            return get_capture_resource(
                ("flash_attn.plan", id(metadata), num_tokens, plan_key),
                lambda: self._build_plan(plan_key, lengths),
            )

        if metadata.flash_attn_lengths is None:
            metadata.flash_attn_lengths = self._build_lengths(metadata, num_tokens)
        if plan_key not in metadata.flash_attn_plans:
            metadata.flash_attn_plans[plan_key] = self._build_plan(plan_key, metadata.flash_attn_lengths)
        return metadata.flash_attn_plans[plan_key]

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        from cann_ops_transformer.ops import flash_attn

        # Retain the model's original scale and output width after zero padding.
        output_head_size = query.shape[-1]
        query = self._pad_head(query)
        metadata = attn_metadata
        plan, attrs = self._get_plan(metadata, query.shape[0], query.shape[-1])

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
            output[: query.shape[0]].copy_(result[..., :output_head_size])
            return output
