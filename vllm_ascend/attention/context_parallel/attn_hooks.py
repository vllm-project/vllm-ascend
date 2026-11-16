#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Per-backend extension hooks for the central attention-metadata assembly.

``build_attn_metadata`` used to dispatch on concrete metadata-builder classes
(``isinstance`` checks in the worker layer). CP backend providers now own that
dispatch: a provider may declare an :class:`AttnMetadataHooks` implementation
and the assembly code only talks to the hook object. The base implementation is
a no-op, so plain builders keep the exact pre-hook behaviour.
"""

from __future__ import annotations

from copy import copy
from typing import Any


class AttnMetadataHooks:
    """Extension points consumed by ``build_attn_metadata``.

    All methods default to the legacy behaviour. A provider overrides only the
    stages its metadata builder needs; the assembly code calls these hooks in
    the same order as the legacy ``isinstance`` branches.
    """

    def prepare_common_metadata(
        self,
        common_attn_metadata,
        *,
        num_reqs: int,
        num_actual_reqs: int,
    ):
        """Prepare the shared metadata before the per-group variants exist."""
        return common_attn_metadata

    def split_variants(
        self,
        attn_group,
        common_attn_metadata,
        *,
        draft_layer_names: set[str] | None,
    ) -> list[tuple[Any, Any]]:
        """Return the ``(layer_names, common_metadata)`` variants to build."""
        return [(attn_group.layer_names, common_attn_metadata)]

    def build_extra_kwargs(self, **ctx: Any) -> dict[str, Any]:
        """Return extra keyword arguments forwarded to the metadata builder."""
        return {}

    def after_build(self, builder, metadata, *, shared: dict[str, Any]) -> None:
        """Update cross-variant state after one metadata object was built."""
        return None


class DSAMetadataHooks(AttnMetadataHooks):
    """DSA: share request-level metadata and inject its graph-mode extras."""

    def build_extra_kwargs(self, **ctx: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_actual_reqs": ctx["num_actual_reqs"],
            "common_ratio_to_sas_metadata": ctx["common_ratio_to_sas_metadata"],
            # FULL graphs replay the padded batch; DSA PCP restore indices must
            # span the padded extent, so capture also implies full-graph mode.
            "full_graph_mode": ctx["full_graph_mode"],
        }
        formatted_slot_mapping = ctx.get("formatted_slot_mapping")
        if formatted_slot_mapping is not None:
            kwargs["formatted_slot_mapping"] = formatted_slot_mapping
        return kwargs

    def after_build(self, builder, metadata, *, shared: dict[str, Any]) -> None:
        # Preserve sharing even if a builder replaces the shared dictionary
        # while constructing its metadata.
        shared["common_ratio_to_sas_metadata"] = builder.common_ratio_to_sas_metadata


class DSAV41MetadataHooks(AttnMetadataHooks):
    """DSA V4.1: inject the compressed-indexer and ring-state extras."""

    def build_extra_kwargs(self, **ctx: Any) -> dict[str, Any]:
        # Only query overrides affect SMLA; unrelated model-specific fields
        # must not disable sharing of batch coordinates or RoPE.
        smla_override = bool(
            {"replay_start", "replay_max_seq_len", "replay_seq_lens", "ori_sparse_indices", "num_query_heads"}
            & ctx["model_specific_extra_kwargs"].keys()
        )
        return {
            "num_actual_reqs": ctx["num_actual_reqs"],
            "skip_ring_state_update": ctx["skip_ring_state_update"],
            "common_v41_metadata": ctx["common_v41_metadata"],
            "common_v41_batch_metadata": ctx["common_v41_batch_metadata"],
            "v41_prefill_override": ctx["v41_prefill_override"],
            "v41_smla_override": smla_override,
            # Upstream capture prepares metadata with runtime mode NONE and
            # for_cudagraph_capture=True. Keep V4.1 indexer branches in the
            # graph even for a short dummy sequence.
            "full_graph_mode": ctx["full_graph_mode"],
        }


class SFAMetadataHooks(AttnMetadataHooks):
    """SFA: split one attention group into target and draft variants."""

    def split_variants(
        self,
        attn_group,
        common_attn_metadata,
        *,
        draft_layer_names: set[str] | None,
    ) -> list[tuple[Any, Any]]:
        variants = [(attn_group.layer_names, common_attn_metadata)]
        if draft_layer_names:
            target_names = [name for name in attn_group.layer_names if name not in draft_layer_names]
            draft_names = [name for name in attn_group.layer_names if name in draft_layer_names]
            if draft_names:
                draft_common = copy(common_attn_metadata)
                draft_common.copy_sfa_draft_index = 0
                draft_common.copy_sfa_restore_tails = False
                variants = ([(target_names, common_attn_metadata)] if target_names else []) + [
                    (draft_names, draft_common)
                ]
        return variants


DSA_METADATA_HOOKS = DSAMetadataHooks()
DSA_V41_METADATA_HOOKS = DSAV41MetadataHooks()
SFA_METADATA_HOOKS = SFAMetadataHooks()
