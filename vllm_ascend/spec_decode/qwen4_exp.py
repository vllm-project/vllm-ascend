# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp speculative decoding for Ascend's hybrid KV cache."""

from copy import copy
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v1.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def is_qwen4_exp_mtp(speculative_config: Any) -> bool:
    """Return whether a speculative config selects Qwen4Exp's native MTP."""
    checker = getattr(speculative_config, "use_qwen4_exp_mtp", None)
    if callable(checker) and checker() is True:
        return True
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    model_architectures = tuple(getattr(draft_model_config, "architectures", ()) or ())
    if "Qwen4ExpMTP" in model_architectures:
        return True
    text_config = getattr(draft_model_config, "hf_text_config", None)
    if text_config is None:
        return False
    text_architectures = tuple(getattr(text_config, "architectures", ()) or ())
    return getattr(text_config, "model_type", None) == "qwen4_exp_mtp" or ("Qwen4ExpMTP" in text_architectures)


class AscendQwen4ExpMTPProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    """Ascend proposer for Qwen4Exp's native, multi-group MTP draft."""

    hidden_size: int

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        AscendSpecDecodeBaseProposer.__init__(
            self,
            vllm_config,
            device,
            True,
            runner=runner,
        )
        qwen_hidden_size = self._get_hidden_size()
        if self.hidden_size != qwen_hidden_size:
            self.hidden_size = qwen_hidden_size
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=device,
            )
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # FULL graph replay records tensor addresses. Keep a persistent slot
        # buffer for every later draft step and cache group.
        self._per_group_next_step_slot_mappings: dict[tuple[int, int], torch.Tensor] = {}

    def _get_hidden_size(self) -> int:
        text_config = self.draft_model_config.hf_text_config
        return int(text_config.hidden_size * text_config.hc_count)

    def model_returns_tuple(self) -> bool:
        return True

    def build_draft_attn_metadata(
        self,
        common_attn_metadata,
        num_input_tokens,
        num_actual_tokens,
    ):
        """Build step-0 metadata with each Qwen cache group's address pair."""
        del num_input_tokens, num_actual_tokens
        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata,
            draft_index=0,
        )
        primary_group = self._get_primary_draft_attn_group()
        primary_metadata = per_layer_attn_metadata[primary_group.layer_names[0]]
        return [per_layer_attn_metadata], primary_metadata

    def _build_cache_only_group_next_step_attn_metadata(
        self,
        common_attn_metadata,
        draft_index,
        num_input_tokens,
        primary_group,
        primary_metadata,
        cache_only_groups,
    ):
        """Rebind later MTP steps to every QSA cache group's address pair."""
        per_layer_attn_metadata = {layer_name: primary_metadata for layer_name in primary_group.layer_names}
        for attn_group in cache_only_groups:
            gid = attn_group.kv_cache_group_id
            group_common = common_attn_metadata
            if gid != self.kv_cache_gid:
                block_table = self._per_group_block_tables.get(gid)
                assert block_table is not None, f"Missing Qwen draft block table for KV cache group {gid}"
                group_common = self.shallow_copy_metadata(common_attn_metadata)
                group_common.block_table_tensor = block_table[: group_common.num_reqs]
                block_size = self._per_group_kernel_block_sizes[gid]
                positions = group_common.positions[: group_common.num_actual_tokens]
                block_numbers = positions // block_size
                block_ids = (
                    block_table[: positions.shape[0]]
                    .gather(
                        dim=1,
                        index=block_numbers.view(-1, 1),
                    )
                    .view(-1)
                )
                slot_mapping = block_ids * block_size + positions % block_size

                key = (draft_index, gid)
                slot_buffer = self._per_group_next_step_slot_mappings.get(key)
                if slot_buffer is None:
                    slot_buffer = torch.empty_like(self.slot_mapping_group[draft_index])
                    self._per_group_next_step_slot_mappings[key] = slot_buffer
                slot_buffer[: slot_mapping.shape[0]].copy_(slot_mapping.to(torch.int32))
                slot_buffer[slot_mapping.shape[0] : num_input_tokens].fill_(-1)
                group_common.slot_mapping = slot_buffer[:num_input_tokens]
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                group_common,
                draft_index,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata

    def _get_draft_slot_block_size(self) -> int:
        """Return the kernel block size of Qwen's main draft cache group."""
        return self.block_size

    def maybe_pad_and_reduce(self, hidden_states, positions):
        return hidden_states, positions

    def maybe_all_gather_and_unpad(
        self,
        last_hidden_states,
        positions,
        hidden_states=None,
    ):
        return last_hidden_states, positions, hidden_states

    def set_per_group_block_table(
        self,
        gid: int,
        block_table: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Keep each hybrid cache group's complete addressing pair."""
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def uses_per_group_kernel_block_sizes(self) -> bool:
        return True

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        common_by_gid: dict[int, CommonAttentionMetadata] = {}
        num_reqs = common_attn_metadata.num_reqs

        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            group_common = common_by_gid.get(gid)
            if group_common is None:
                if gid == self.kv_cache_gid:
                    group_common = common_attn_metadata
                else:
                    block_table = self._per_group_block_tables.get(gid)
                    assert block_table is not None, f"Missing Qwen draft block table for KV cache group {gid}"
                    slot_mapping = self._per_group_slot_mappings.get(gid)
                    assert slot_mapping is not None, f"Missing Qwen draft slot mapping for KV cache group {gid}"
                    group_common = copy(common_attn_metadata)
                    group_common.block_table_tensor = block_table[:num_reqs]
                    group_common.slot_mapping = slot_mapping[: group_common.num_actual_tokens]
                common_by_gid[gid] = group_common

            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=group_common,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        return per_group_attn_metadata, per_layer_attn_metadata

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        num_mtp_layers = self.draft_model_config.hf_text_config.mtp_num_hidden_layers
        if num_mtp_layers != 1:
            raise NotImplementedError("Qwen4Exp MTP proposer only supports one MTP layer")
        assert kernel_block_sizes is not None, "Qwen MTP requires resolved kernel block sizes"
        assert len(kernel_block_sizes) == len(kv_cache_config.kv_cache_groups), (
            "Qwen MTP requires one kernel block size per KV cache group"
        )
        self._per_group_kernel_block_sizes = {gid: block_size for gid, block_size in enumerate(kernel_block_sizes)}

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        layer_to_gid, layer_to_spec = self._map_draft_layers_to_groups(kv_cache_config)
        main_layers = [name for name, spec in layer_to_spec.items() if type(spec) is FullAttentionSpec]
        assert len(main_layers) == 1, "Qwen4Exp MTP requires exactly one main cache owner"
        self.kv_cache_gid = layer_to_gid[main_layers[0]]

        attention_groups: list[AttentionGroup] = []
        for layer_name in sorted(self._draft_attn_layer_names):
            attn_layer = all_attn_layers[layer_name]
            gid = layer_to_gid[layer_name]
            attn_group = AttentionGroup(
                backend=attn_layer.get_attn_backend(),
                layer_names=[layer_name],
                kv_cache_spec=layer_to_spec[layer_name],
                kv_cache_group_id=gid,
            )
            attn_group.create_metadata_builders(
                self.vllm_config,
                self.device,
                kernel_block_size=kernel_block_sizes[gid],
            )
            attention_groups.append(attn_group)

        self.draft_attn_groups = sorted(
            attention_groups,
            key=lambda group: (
                group.kv_cache_group_id != self.kv_cache_gid,
                group.kv_cache_group_id,
                group.backend.full_cls_name(),
                group.layer_names[0],
            ),
        )
        self.block_size = kernel_block_sizes[self.kv_cache_gid]

    def _map_draft_layers_to_groups(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> tuple[dict[str, int], dict[str, KVCacheSpec]]:
        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name not in self._draft_attn_layer_names:
                    continue
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    spec = group_spec.kv_cache_specs.get(layer_name)
                else:
                    # Hybrid KV grouping may emit a direct homogeneous spec
                    # alongside packed UniformType groups. The direct group
                    # spec is already the per-layer spec.
                    spec = group_spec
                assert spec is not None, f"Qwen draft cache group {gid} has no spec for {layer_name}"
                layer_to_gid[layer_name] = gid
                layer_to_spec[layer_name] = spec

        assert layer_to_spec.keys() == self._draft_attn_layer_names, (
            "Qwen draft KV cache configuration is missing layers: "
            f"{sorted(self._draft_attn_layer_names - layer_to_spec.keys())}"
        )
        return layer_to_gid, layer_to_spec


__all__ = ["AscendQwen4ExpMTPProposer", "is_qwen4_exp_mtp"]
