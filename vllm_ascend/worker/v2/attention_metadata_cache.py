# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata

from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder


class AttentionMetadataCache:
    """Share templates within one build. Each group supplies its own block table and slots."""

    def __init__(self, provider: ModelSpecificAttnMetadata | None, num_reqs: int, for_capture: bool):
        self.provider = provider
        self.num_reqs = num_reqs
        self.for_capture = for_capture
        self.templates: dict[tuple, tuple[Any, dict]] = {}
        self._graph_updates: list[tuple[Any, list]] = []
        self._common_kwargs: dict | None = None
        self._native_common = type(provider) is MambaHybridAttnMetadata and (
            getattr(provider.get_extra_common_attn_kwargs, "__func__", None)
            is MambaHybridAttnMetadata.get_extra_common_attn_kwargs
        )
        # An override of either provider hook can supply group-local tensors.
        self._batch_global = provider is None or (
            self._native_common
            and getattr(provider.get_extra_attn_kwargs, "__func__", None)
            is MambaHybridAttnMetadata.get_extra_attn_kwargs
        )

    def common_kwargs(self, group_id: int) -> dict:
        if self.provider is None:
            return {}
        if not self._native_common:
            return self.provider.get_extra_common_attn_kwargs(group_id, self.num_reqs)
        if self._common_kwargs is None:
            self._common_kwargs = self.provider.get_extra_common_attn_kwargs(group_id, self.num_reqs)
        return self._common_kwargs.copy()  # The caller consumes is_prefilling with pop().

    def key_for(self, builder: Any, common: AscendCommonAttentionMetadata, common_kwargs: dict) -> tuple | None:
        eligible = self._batch_global and common.pcp_context is None and not common_kwargs
        if isinstance(builder, AscendGDNAttentionMetadataBuilder):
            eligible = eligible and builder.supports_update_block_table and isinstance(builder.kv_cache_spec, MambaSpec)
            # Replay must refresh the same buffers that capture recorded.
            if self.for_capture:
                builder._gdn_reuse_at_capture = eligible
            else:
                captured = getattr(builder, "_gdn_reuse_at_capture", None)
                if captured is False:
                    return None
                if captured is True and not eligible:
                    raise ValueError("Captured GDN metadata reuse requires a batch-global Mamba provider")
            if eligible:
                return builder.kv_cache_spec, type(builder)
        elif eligible and not self.for_capture and type(builder) is AscendAttentionMetadataBuilder:
            if builder.supports_update_block_table:
                return builder.kv_cache_spec, builder.decode_threshold, common.causal
        return None

    def store(self, key: tuple, metadata: Any, builder: Any) -> None:
        update_kwargs = {}
        updater = getattr(builder, "graph_state_updater", None)
        if updater is not None:
            updates: list = []
            self._graph_updates.append((updater, updates))
            update_kwargs["graph_state_updates"] = updates
        self.templates[key] = metadata, update_kwargs

    def apply_graph_updates(self) -> None:
        for updater, updates in self._graph_updates:
            updater.apply(updates)
