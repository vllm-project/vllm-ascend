# SPDX-License-Identifier: Apache-2.0
"""Gemma4 MTP proposer for Ascend NPUs."""

import copy

from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendGemma4Proposer(Gemma4Proposer, AscendSpecDecodeBaseProposer):
    """Reuse vLLM's Gemma4 proposer with Ascend execution support."""

    def _setup_centroids_cuda_graphs(self) -> None:
        # Centroid sampling runs eagerly on NPU; ACL graph capture is handled by
        # AscendSpecDecodeBaseProposer.
        self._centroids_sizes: list[int] = []

    def _maybe_share_lm_head(self, target_language_model) -> None:
        # Gemma4 keeps its draft-dimension lm_head. The Ascend implementation
        # still needs to install the ACL graph wrapper.
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        self.supports_mm_inputs = False
        self._sync_kv_sharing_target_to_impl()

    def _sync_kv_sharing_target_to_impl(self) -> None:
        """Propagate late-bound KV-sharing targets to Ascend backends."""
        attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        for layer_name in self._draft_attn_layer_names:
            attn = attn_layers[layer_name]
            target = getattr(attn, "kv_sharing_target_layer_name", None)
            impl = getattr(attn, "impl", None)
            if target is not None and impl is not None:
                impl.kv_sharing_target_layer_name = target

    def _get_group_common_attn_metadata(self, attn_group, common_attn_metadata):
        block_table = self._per_group_block_tables.get(attn_group.kv_cache_group_id)
        if block_table is None:
            return common_attn_metadata
        group_metadata = copy.copy(common_attn_metadata)
        group_metadata.block_table_tensor = block_table[: common_attn_metadata.num_reqs]
        return group_metadata

    def _build_multi_group_graph_capture_metadata(self, common_attn_metadata, draft_index):
        per_layer_attn_metadata = {}
        for attn_group in self.draft_attn_groups:
            group_metadata = self._get_group_common_attn_metadata(
                attn_group,
                common_attn_metadata,
            )
            attn_metadata = attn_group.get_metadata_builder().build_for_graph_capture(
                group_metadata,
                AscendAttentionState.SpecDecoding,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata

    def _get_attn_metadata_layer_names(self, attn_group):
        return attn_group.layer_names

    def attn_update_stack_num_spec_norm(
        self,
        draft_index,
        old_common_metadata,
        *args,
        attn_group=None,
        **kwargs,
    ):
        assert attn_group is not None
        group_metadata = self._get_group_common_attn_metadata(
            attn_group,
            old_common_metadata,
        )
        kwargs["attn_group"] = attn_group
        return super().attn_update_stack_num_spec_norm(
            draft_index,
            group_metadata,
            *args,
            **kwargs,
        )

    def build_draft_attn_metadata(
        self,
        common_attn_metadata,
        num_input_tokens,
        num_actual_tokens,
    ):
        per_layer_attn_metadata = {}
        for attn_group in self.draft_attn_groups:
            group_metadata = self._get_group_common_attn_metadata(
                attn_group,
                common_attn_metadata,
            )
            attn_metadata = attn_group.get_metadata_builder().build(
                0,
                group_metadata,
                self.runner.get_model(),
            )
            attn_metadata.attn_state = AscendAttentionState.SpecDecoding
            if hasattr(attn_metadata, "causal") and not attn_metadata.causal:
                attn_metadata.attn_mask = None
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        attn_metadata = per_layer_attn_metadata[self.draft_attn_groups[0].layer_names[0]]
        return [per_layer_attn_metadata], attn_metadata
