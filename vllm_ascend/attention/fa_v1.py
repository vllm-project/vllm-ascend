# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Device-length CANN FlashAttn for A5 V2 drafts."""

from dataclasses import dataclass, replace
from typing import cast

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window


@dataclass
class AscendFlashAttentionMetadata(AscendMetadata):
    flash_attn_mask: torch.Tensor | None = None
    flash_attn_plan: tuple[torch.Tensor, dict] | None = None


class AscendFlashAttentionMetadataBuilder(AscendAttentionMetadataBuilder):
    metadata_cls = AscendFlashAttentionMetadata
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
        impl = cast(AscendFlashAttentionImpl, layers[layer_names[0]].impl)
        self.num_heads = impl.num_heads
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_size = AscendFlashAttentionBackend.get_padded_head_size(kv_cache_spec.head_size)
        self.sliding_window = impl.sliding_window
        self.layout_kv = "PA_BNBD" if impl.use_bnsd_kv_cache else "PA_BBND"
        # A merged full KV cache spec may include both full and windowed layers.
        if any(
            cast(AscendFlashAttentionImpl, layer.impl).sliding_window != self.sliding_window
            for layer in layers.values()
        ):
            raise ValueError("CANN FlashAttn requires the same sliding window within each attention group.")

        self.plan_buffer: torch.Tensor | None = None
        if vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            from cann_ops_transformer.ops import flash_attn_metadata

            max_reqs = vllm_config.scheduler_config.max_num_seqs
            # Use CANN's Meta implementation to size the buffer before capture.
            plan_meta = flash_attn_metadata(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                cu_seqlens_q=torch.empty(max_reqs + 1, dtype=torch.int32, device="meta"),
                seqused_kv=torch.empty(max_reqs, dtype=torch.int32, device="meta"),
                layout_q="TND",
                layout_kv=self.layout_kv,
                layout_out="TND",
            )
            self.plan_buffer = torch.empty_like(plan_meta, device=device)

    def _get_flash_attn_plan(self, metadata: AscendFlashAttentionMetadata) -> tuple[torch.Tensor, dict]:
        from cann_ops_transformer.ops import flash_attn_metadata

        num_reqs = metadata.block_tables.shape[0]
        boundaries = metadata.query_start_loc_gpu[: num_reqs + 1].int()
        # Dummy requests may have repeated or negative query boundaries.
        boundaries = boundaries.clamp(min=0, max=metadata.num_actual_tokens).cummax(dim=0).values
        lengths = metadata.seq_lens_gpu[:num_reqs].int().clamp(min=1)
        sliding_window = self.sliding_window
        window = sliding_window - 1 if sliding_window is not None else -1
        # Both CANN operators require the same lengths, mask and layout.
        # Mask modes: 0 = unmasked, 3 = causal, 4 = sliding window.
        attrs = dict(
            cu_seqlens_q=boundaries,
            seqused_kv=lengths,
            max_seqlen_q=metadata.max_query_len,
            max_seqlen_kv=-1,
            mask_mode=4 if sliding_window is not None else 3 if metadata.causal else 0,
            win_left=window,
            win_right=0 if metadata.causal and sliding_window is not None else window,
            layout_q="TND",
            layout_kv=self.layout_kv,
            layout_out="TND",
        )
        plan = flash_attn_metadata(self.num_heads, self.num_kv_heads, self.head_size, **attrs)
        return plan, attrs

    def _store_flash_attn_plan(self, plan: torch.Tensor) -> torch.Tensor:
        if self.plan_buffer is None:
            return plan
        stored_plan = self.plan_buffer[: plan.numel()]
        stored_plan.copy_(plan)
        return stored_plan

    def update_draft_decode_metadata(self, metadata: AscendFlashAttentionMetadata) -> None:
        plan, attrs = self._get_flash_attn_plan(metadata)
        metadata.flash_attn_plan = self._store_flash_attn_plan(plan), attrs

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = self._build_metadata(common_attn_metadata)
        self.update_draft_decode_metadata(metadata)
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # The capture callback prepares plans inside the graph from the live buffers.
        return self._build_metadata(common_attn_metadata)

    def _build_metadata(self, common_attn_metadata):
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

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        from cann_ops_transformer.ops import flash_attn

        # Retain the model's original scale and output width after zero padding.
        output_head_size = query.shape[-1]
        query = self._pad_head(query)
        metadata = attn_metadata
        assert metadata.flash_attn_plan is not None
        plan, attrs = metadata.flash_attn_plan

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
