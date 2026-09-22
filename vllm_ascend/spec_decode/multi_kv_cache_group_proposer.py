# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

_MULTI_KV_CACHE_GROUP_MTP_MODEL_TYPES = {"glm5_next_mtp"}
_MULTI_KV_CACHE_GROUP_MTP_ARCHITECTURES = {"Glm5NextMTPModel"}


def is_multi_kv_cache_group_mtp(vllm_config: VllmConfig) -> bool:
    """Enable selector for the multi-KV-cache-group MTP proposer.

    This is a module-level helper, not an inherited method. The factory calls
    it before constructing a proposer. Add future supported MTP model types or
    architectures to the allowlists above.
    """
    speculative_config = vllm_config.speculative_config
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    architectures = set(getattr(hf_config, "architectures", None) or ())
    return model_type in _MULTI_KV_CACHE_GROUP_MTP_MODEL_TYPES or bool(
        architectures & _MULTI_KV_CACHE_GROUP_MTP_ARCHITECTURES
    )


class AscendMultiKVCacheGroupMTPProposer(AscendEagleProposer):
    """MTP proposer for draft layers split across physical KV cache groups.

    The constructor is intentionally inherited unchanged from
    ``AscendEagleProposer``. Only eager runtime metadata paths that need a
    physical KV-cache-group view are overridden below.
    """

    def _get_draft_layer_kv_cache_groups(self, kv_cache_config: KVCacheConfig) -> dict[str, int]:
        """Helper: map each draft attention layer to its physical group id.

        This method is new in this subclass. It validates that every draft
        layer discovered while loading the model appears in ``KVCacheConfig``.
        """
        layer_to_group = {
            layer_name: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
            if layer_name in self._draft_attn_layer_names
        }
        missing_layers = self._draft_attn_layer_names - layer_to_group.keys()
        if missing_layers:
            raise ValueError(f"Drafting layers are missing from the KV cache config: {sorted(missing_layers)}")
        return layer_to_group

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Override the inherited attention-backend initializer.

        ``AscendSpecDecodeBaseProposer`` inherits the default implementation
        from vLLM's ``SpecDecodeBaseProposer``, which requires all draft layers
        to share one KV cache group. This override creates one ``AttentionGroup``
        per physical cache group. The executable attention group becomes the
        primary group; cache-only groups retain their own logical block size.
        A single-group configuration delegates unchanged to ``super()``.
        """
        layer_to_group = self._get_draft_layer_kv_cache_groups(kv_cache_config)
        draft_group_ids = set(layer_to_group.values())
        self._uses_multi_group_kv_cache = len(draft_group_ids) > 1
        if not self._uses_multi_group_kv_cache:
            super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
            return

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        attention_groups: dict[tuple[str, int], AttentionGroup] = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            gid = layer_to_group[layer_name]
            group_kv_cache_spec = kv_cache_config.kv_cache_groups[gid].kv_cache_spec
            layer_kv_cache_spec = group_kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = group_kv_cache_spec.kv_cache_specs[layer_name]

            attn_backend = all_attn_layers[layer_name].get_attn_backend()
            backend_key = (attn_backend.full_cls_name(), gid)
            if backend_key in attention_groups:
                attention_groups[backend_key].layer_names.append(layer_name)
                continue

            attn_group = AttentionGroup(
                backend=attn_backend,
                layer_names=[layer_name],
                kv_cache_spec=layer_kv_cache_spec,
                kv_cache_group_id=gid,
            )
            kernel_block_size = (
                kernel_block_sizes[gid]
                if not self._is_cache_only_draft_attn_group(attn_group)
                and kernel_block_sizes is not None
                and gid < len(kernel_block_sizes)
                else None
            )
            attn_group.create_metadata_builders(
                self.vllm_config,
                self.device,
                kernel_block_size=kernel_block_size,
            )
            attention_groups[backend_key] = attn_group

        if not attention_groups:
            raise ValueError("No KV cache group found for drafting layers")

        self.draft_attn_groups = list(attention_groups.values())
        primary_group = self._get_primary_draft_attn_group()
        self.kv_cache_gid = primary_group.kv_cache_group_id
        if kernel_block_sizes is not None and self.kv_cache_gid < len(kernel_block_sizes):
            self.block_size = kernel_block_sizes[self.kv_cache_gid]
        else:
            self.block_size = primary_group.get_metadata_builder().kv_cache_spec.block_size

        logger.info(
            "Initialized GLM5-Next drafting attention groups for KV cache group ids %s; primary group id is %d",
            sorted(draft_group_ids),
            self.kv_cache_gid,
        )

    def _draft_block_table_width(self, attn_group: AttentionGroup) -> int:
        """Helper: return the block-table width consumed by one group.

        This method is new in this subclass. It converts the maximum logical
        sequence length into the kernel block-table width when logical and
        physical block sizes differ, as they do for the indexer cache.
        """
        builder = attn_group.get_metadata_builder()
        logical_block_size = getattr(builder, "logical_block_size", None)
        blocks_per_logical_block = getattr(builder, "kernel_blocks_per_logical_block", None)
        if (
            isinstance(logical_block_size, int)
            and logical_block_size > 0
            and isinstance(blocks_per_logical_block, int)
            and blocks_per_logical_block > 0
        ):
            max_logical_blocks = (self.max_model_len + logical_block_size - 1) // logical_block_size
            return max_logical_blocks * blocks_per_logical_block
        return builder.kv_cache_spec.max_num_blocks_per_req(self.vllm_config, self.max_model_len)

    def _common_attn_metadata_for_draft_group(
        self,
        common_attn_metadata,
        attn_group,
        num_input_tokens,
    ):
        """Override the runtime common-metadata view hook for one group.

        The base hook returns the common metadata unchanged. This override
        crops every group's block table to the width consumed by its builder.
        For a secondary group it computes every input token's slot from the
        device positions, query boundaries, and that group's block table. The input
        metadata is shallow-copied so the primary group's view is not
        overwritten.
        """
        if not self._uses_multi_group_kv_cache:
            return common_attn_metadata

        gid = attn_group.kv_cache_group_id
        group_metadata = copy.copy(common_attn_metadata)
        if gid == self.kv_cache_gid:
            block_table_tensor = common_attn_metadata.block_table_tensor
        else:
            block_table = self.runner.input_batch.block_table[gid]
            num_reqs = group_metadata.num_reqs
            num_actual_tokens = common_attn_metadata.num_actual_tokens
            block_table.compute_slot_mapping(
                num_reqs,
                common_attn_metadata.query_start_loc,
                common_attn_metadata.positions[:num_actual_tokens],
            )
            block_table_tensor = block_table.get_device_tensor()[:num_reqs]
            # All draft steps are built before any forward. The block table's
            # scratch mapping is overwritten when the next step is prepared.
            group_slot_mapping = block_table.slot_mapping.gpu[:num_input_tokens].clone()
            group_slot_mapping[num_actual_tokens:].fill_(PADDING_SLOT_ID)
            group_metadata.slot_mapping = group_slot_mapping

        group_metadata.block_table_tensor = block_table_tensor[
            : group_metadata.num_reqs, : self._draft_block_table_width(attn_group)
        ]
        return group_metadata

    @staticmethod
    def _copy_cache_only_draft_metadata(attn_metadata):
        """Helper: preserve a step's tensors before a builder reuses its buffers.

        KPool builders use persistent buffers for target graph replay. Draft
        metadata for every step must coexist until the merged draft finishes,
        including step 0's multi-token query layout and compressed cache slots.
        """
        if not is_dataclass(attn_metadata) or isinstance(attn_metadata, type):
            return attn_metadata
        return replace(
            attn_metadata,
            **{
                field.name: value.clone()
                for field in fields(attn_metadata)
                if isinstance(value := getattr(attn_metadata, field.name), torch.Tensor)
            },
        )

    def attn_update_stack_num_spec_norm(
        self,
        draft_index,
        old_common_metadata,
        batch_size,
        input_batch_size,
        used_update_positions,
        aclgraph_runtime_mode,
        ori_seq_len=None,
        ori_seq_len_cpu=None,
        slot_indices=None,
        mtp_slot_mapping=None,
        attn_group=None,
    ):
        """Override the transition from verification rows to one row per request.

        The first pass may retain rejected tokens as padding. Later draft steps
        start at each request's selected token, so reset lengths to its actual
        device position before the inherited updater advances the step once.
        CPU lengths still describe the optimistic verification batch and must
        be invalidated without introducing a device-to-host synchronization.
        """
        if self._uses_multi_group_kv_cache and draft_index == 1:
            old_common_metadata = copy.copy(old_common_metadata)
            old_common_metadata.seq_lens = old_common_metadata.seq_lens.clone()
            old_common_metadata.seq_lens[:batch_size] = used_update_positions + 1
            old_common_metadata.seq_lens_cpu = None
            old_common_metadata._seq_lens_cpu = None
            old_common_metadata.num_computed_tokens_cpu = None
            old_common_metadata._num_computed_tokens_cpu = None
        return super().attn_update_stack_num_spec_norm(
            draft_index,
            old_common_metadata,
            batch_size,
            input_batch_size,
            used_update_positions,
            aclgraph_runtime_mode,
            ori_seq_len=ori_seq_len,
            ori_seq_len_cpu=ori_seq_len_cpu,
            slot_indices=slot_indices,
            mtp_slot_mapping=mtp_slot_mapping,
            attn_group=attn_group,
        )

    def build_draft_attn_metadata(
        self,
        common_attn_metadata,
        num_input_tokens,
        num_actual_tokens,
    ):
        """Override step-0 draft attention metadata construction.

        The base implementation builds every attention group from one common
        block table and slot mapping. This override supplies a physical-group
        view to each builder, then returns the executable primary group's
        metadata as the shared MTP state. Single-group models delegate to the
        inherited implementation.
        """
        if not self._uses_multi_group_kv_cache:
            return super().build_draft_attn_metadata(
                common_attn_metadata,
                num_input_tokens,
                num_actual_tokens,
            )

        per_layer_attn_metadata: dict[str, Any] = {}
        shared_draft_cache: dict[str, Any] = dict(common_ratio_to_sas_metadata=dict()) if self.use_compress else {}
        for attn_group in self.draft_attn_groups:
            extra_attn_metadata_args = dict(shared_draft_cache)
            if self.use_compress:
                extra_attn_metadata_args["block_size"] = attn_group.kv_cache_spec.block_size
            attn_metadata = attn_group.get_metadata_builder().build(
                0,
                self._common_attn_metadata_for_draft_group(
                    common_attn_metadata,
                    attn_group,
                    num_input_tokens,
                ),
                self.runner.get_model(),
                **extra_attn_metadata_args,
            )
            if hasattr(attn_metadata, "causal") and not attn_metadata.causal:
                attn_metadata.attn_mask = None
            if self._is_cache_only_draft_attn_group(attn_group):
                attn_metadata = self._copy_cache_only_draft_metadata(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

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
        """Override cache-only metadata construction for later MTP steps.

        ``attn_update_stack_num_spec_norm`` has already advanced shared state
        once for the primary group before this hook is called. This override
        rebinds that updated logical state to each cache-only group's physical
        cache without advancing sequence lengths, positions, or DCP metadata a
        second time. Single-group models use the base hook unchanged.
        """
        if not self._uses_multi_group_kv_cache:
            return super()._build_cache_only_group_next_step_attn_metadata(
                common_attn_metadata,
                draft_index,
                num_input_tokens,
                primary_group,
                primary_metadata,
                cache_only_groups,
            )

        per_layer_attn_metadata: dict[str, Any] = {
            layer_name: primary_metadata for layer_name in primary_group.layer_names
        }
        for attn_group in cache_only_groups:
            group_common_attn_metadata = self._common_attn_metadata_for_draft_group(
                common_attn_metadata,
                attn_group,
                num_input_tokens,
            )
            extra_attn_metadata_args: dict[str, Any] = {}
            if self.use_compress:
                extra_attn_metadata_args["common_ratio_to_sas_metadata"] = {}
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                group_common_attn_metadata,
                draft_index,
                **extra_attn_metadata_args,
            )
            attn_metadata = self._copy_cache_only_draft_metadata(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata
