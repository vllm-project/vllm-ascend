# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer and group interfaces for the two supported vLLM cache planners."""

from dataclasses import replace

from vllm.v1 import kv_cache_interface
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    from vllm.utils.math_utils import cdiv
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec as CacheGroup
    from vllm.v1.kv_cache_interface import KVCacheSpec as CacheRequirement
    from vllm.v1.kv_cache_interface import (
        SlidingWindowMLASpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
        iter_layer_specs,
    )
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    MLA_LAYOUT_TYPES = (MLAAttentionSpec, SlidingWindowMLASpec)
else:
    from vllm.v1.core.kv_cache_planning import can_share_block_table
    from vllm.v1.kv_cache_interface import KVCacheGroup as CacheGroup
    from vllm.v1.kv_cache_interface import KVCacheLayerConfig as CacheRequirement

    MLA_LAYOUT_TYPES = (MLAAttentionSpec,)


def with_cache_layouts(configs, specs):
    if vllm_version_is("0.33.0"):
        return specs
    return {name: replace(configs[name], spec=spec) for name, spec in specs.items()}


def make_cache_group(configs, specs, *, uniform=True):
    if not vllm_version_is("0.33.0"):
        members = with_cache_layouts(configs, specs)
        if not can_share_block_table(members):
            return None
        return CacheGroup.from_layers(members)
    if uniform:
        spec = UniformTypeKVCacheSpecs.from_specs(specs)
    else:
        first = next(iter(specs.values()))
        spec = first.merge(list(specs.values()))
    return CacheGroup(list(specs), spec) if spec is not None else None


def group_max_blocks(group, config):
    if vllm_version_is("0.33.0"):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return spec.max_memory_usage_pages(config)
        return cdiv(spec.max_memory_usage_bytes(config), spec.page_size_bytes)
    return group.max_allocated_blocks_per_request(config)


def get_layer_spec(layer, vllm_config):
    if vllm_version_is("0.33.0"):
        return layer.get_kv_cache_spec(vllm_config)
    config = layer.get_kv_cache_config(vllm_config)
    return config.spec if config is not None else None


def group_layer_specs(group):
    if vllm_version_is("0.33.0"):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            # A release group may list shared consumers alongside the local cache owners.
            return {name: spec.kv_cache_specs[name] for name in group.layer_names if name in spec.kv_cache_specs}
        return dict.fromkeys(group.layer_names, spec)
    return group.kv_cache_specs


def full_attention_spec_type():
    """Physical spec class used by a full-attention layer on this vLLM version."""
    if vllm_version_is("0.33.0"):
        return kv_cache_interface.FullAttentionSpec
    return kv_cache_interface.AttentionSpec


def get_cache_requirements(worker):
    if vllm_version_is("0.33.0"):
        return worker.get_kv_cache_spec()
    return worker.get_kv_cache_layer_configs()


def is_plain_full_attention(spec, config):
    if vllm_version_is("0.33.0"):
        return type(spec) is kv_cache_interface.FullAttentionSpec
    return type(spec) is kv_cache_interface.AttentionSpec and config.manager_cls is FullAttentionManager


def group_params(group):
    if vllm_version_is("0.33.0"):
        return group_distinct_specs(group)[0]
    return group.manager_config.params


def _release_manager_for_spec(spec):
    manager = KVCacheSpecRegistry.get_manager_class(spec)
    if manager is None:
        from vllm.v1.core import single_type_kv_cache_manager

        manager = getattr(single_type_kv_cache_manager, "spec_manager_map", {}).get(type(spec))
    return manager


def group_manager(group, *, required=False):
    if vllm_version_is("0.33.0"):
        spec = group_params(group)
        manager = _release_manager_for_spec(spec)
        if manager is None and required:
            raise AssertionError(f"No manager registered for KVCacheSpec {type(spec)}")
        return manager
    return group.manager_cls


def group_prefix_cacheable(group):
    if vllm_version_is("0.33.0"):
        from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable

        return is_prefix_cacheable(group.kv_cache_spec)
    return group.prefix_cacheable


def group_has_sliding_window(group):
    if vllm_version_is("0.33.0"):
        return any(isinstance(spec, SlidingWindowSpec) for spec in group_distinct_specs(group))
    return issubclass(group.manager_cls, SlidingWindowManager)


def group_sliding_reach(group):
    if vllm_version_is("0.33.0"):
        return group_params(group).num_retained_tokens
    return group.sliding_window_reach


def group_distinct_specs(group):
    if vllm_version_is("0.33.0"):
        return tuple(dict.fromkeys(iter_layer_specs(group.kv_cache_spec)))
    return group.distinct_specs


def _block_table_policy(group):
    if vllm_version_is("0.33.0"):
        return group.kv_cache_spec
    return group


def group_uses_slot_mapping(group):
    return _block_table_policy(group).uses_slot_mapping


def group_dcp_sharded(group):
    return _block_table_policy(group).dcp_sharded


def group_dcp_block_size(group, dcp_world_size):
    from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size

    return resolve_dcp_kv_block_size(_block_table_policy(group), dcp_world_size)


def group_block_table_width(group, config, max_model_len):
    if vllm_version_is("0.33.0"):
        from vllm.v1.worker.block_table import get_block_table_width

        spec = group.kv_cache_spec
        width = spec.max_num_blocks_per_req(config, max_model_len)
        return get_block_table_width(width, spec.block_size, token_alignment=spec.block_table_token_alignment)
    return group.block_table_width(config, max_model_len)


def group_block_size(group):
    return _block_table_policy(group).block_size


def cache_layouts(requirements: dict[str, CacheRequirement]) -> dict[str, kv_cache_interface.KVCacheSpec]:
    if not vllm_version_is("0.33.0"):
        return {name: config.spec for name, config in requirements.items()}
    return requirements


def configure_cache_transfer(vllm_config, kv_cache_config):
    kv_cache_config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import draft_only_group_ids

        # Release scheduler conversion keeps only the first spec in a uniform
        # group, losing both residency and per-layer is_draft ownership. Keep
        # the PD context and external prefix-store classifications separate:
        # a mixed group needs draft block IDs AND ordinary target hit keys.
        # Remove these snapshots once upstream preserves per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        kv_cache_config.dspark_draft_only_group_ids = (
            draft_only_group_ids(kv_cache_config.kv_cache_groups)
            if vllm_version_is("0.33.0")
            else vllm_config.dspark_draft_only_group_ids
        )
