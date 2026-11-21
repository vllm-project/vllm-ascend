# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan layerwise cache reuse from native cache groups."""

from collections import defaultdict
from math import lcm
from types import MappingProxyType
from typing import Any

from vllm.config import VllmConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, KVCacheTensor, MambaSpec

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import get_backend_class

from .layerwise_cache_layout import (
    LayerwiseCacheConfig,
    LayerwiseReuseLayout,
    SpecReuseRules,
    cache_view_alignment,
    get_cache_views,
    get_layerwise_kv_cache_specs,
    get_physical_layer_index,
)

SPEC_REUSE_RULES = MappingProxyType(
    {
        "glm5_next_text": "vllm_ascend.models.glm5next.cache_reuse.GLM5_NEXT_SPEC_REUSE_RULES",
        "glm_moe_dsa": "vllm_ascend.models.glm5next.cache_reuse.GLM_DSA_SPEC_REUSE_RULES",
    }
)


def get_spec_reuse_rules(vllm_config: VllmConfig) -> SpecReuseRules | None:
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    model_type = getattr(text_config, "model_type", None)
    if not isinstance(model_type, str):
        return None
    path = SPEC_REUSE_RULES.get(model_type)
    return resolve_obj_by_qualname(path) if path is not None else None


def get_layerwise_ascend_store_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the extra config for layerwise AscendStore."""
    store_connectors = ("AscendStoreConnector", "MooncakeConnectorStoreV1")
    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    if connector_name == "MultiConnector":
        for child_config in extra_config.get("connectors", ()):
            if isinstance(child_config, dict) and child_config.get("kv_connector") in store_connectors:
                extra_config = child_config.get("kv_connector_extra_config") or {}
                break
        else:
            return None
    elif connector_name not in store_connectors:
        return None

    if extra_config.get("use_layerwise", False):
        return extra_config
    return None


def get_layerwise_gva_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the config for a layerwise store using GVA."""
    extra_config = get_layerwise_ascend_store_config(kv_transfer_config)
    if extra_config is None:
        return None
    backend_class = get_backend_class(str(extra_config.get("backend", "mooncake")))
    if backend_class is not None and backend_class.layerwise_data_plane == "gva":
        return extra_config
    return None


def get_layerwise_offload_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Find a layerwise store with an explicit shared-buffer count."""
    # TODO(bowgneo): Mooncake support for shared-buffer offload is pending. Detect the
    # request from config regardless of backend.
    extra_config = get_layerwise_ascend_store_config(kv_transfer_config)
    if extra_config is not None and extra_config.get("layerwise_num_shared_buffers") is not None:
        return extra_config
    return None


class LayerwiseCachePlanner:
    @staticmethod
    def can_reuse_spec(spec: KVCacheSpec, rules: SpecReuseRules | None) -> bool:
        """Apply the rule for the concrete spec type, defaulting to false."""
        rule = False if rules is None else rules.get(type(spec), False)
        result = rule(spec) if callable(rule) else rule
        if not isinstance(result, bool):
            raise TypeError("Spec reuse rules must return bool")
        return result

    def __init__(
        self,
        config: KVCacheConfig,
        vllm_config: VllmConfig,
        extra_config: dict[str, Any],
        *,
        spec_reuse_rules: SpecReuseRules,
    ) -> None:
        self.config = config
        self.extra_config = extra_config
        self.spec_reuse_rules = spec_reuse_rules
        self.spec_by_entry = get_layerwise_kv_cache_specs(config)
        self.group_id_by_entry = {
            entry_name: group_id
            for group_id, group in enumerate(config.kv_cache_groups)
            for entry_name in group.layer_names
        }
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.layer_id_by_entry = {
            entry_name: get_physical_layer_index(entry_name, self.total_layers) for entry_name in self.spec_by_entry
        }
        self.draft_names = {
            name
            for name, spec in self.spec_by_entry.items()
            if getattr(spec, "is_draft", False) or self.layer_id_by_entry[name] >= self.total_layers
        }
        self.target_layer_ids = tuple(
            sorted({index for name, index in self.layer_id_by_entry.items() if name not in self.draft_names})
        )

    def build(self) -> LayerwiseReuseLayout:
        settings = LayerwiseCacheConfig.parse(self.total_layers, self.extra_config, stage_layers=self.target_layer_ids)
        independent = settings.independent_layers
        num_shared_buffers = settings.num_shared_buffers or 0
        native_cache_tensors = self.config.kv_cache_tensors
        num_blocks = self.config.num_blocks
        # Block stride in bytes -> cache entry names eligible for rotation.
        reusable_entries_by_stride: dict[int, list[str]] = defaultdict(list)
        # Cache entry name -> length of its native cache view in bytes.
        cache_view_sizes: dict[str, int] = {}
        # Each slot is (byte offset in backing, block stride in bytes, entry names).
        slot_layouts: list[tuple[int, int, list[str]]] = []
        backing_bytes = 0
        reusable_entry_names = set()
        if num_shared_buffers:
            reusable_entry_names = {
                entry_name
                for group in self.config.kv_cache_groups
                if is_prefix_cacheable(group.kv_cache_spec)
                for entry_name in group.layer_names
                if entry_name not in self.draft_names
                and self.layer_id_by_entry[entry_name] not in independent
                and self.can_reuse_spec(self.spec_by_entry[entry_name], self.spec_reuse_rules)
            }
        if reusable_entry_names:
            native_cache_views = get_cache_views(self.config, self.spec_by_entry)
            cache_view_sizes = {
                entry_name: view_end - view_start
                for entry_name, (_, view_start, view_end) in native_cache_views.items()
            }
            # Block stride in bytes -> resident slots, each a list of cache entry names.
            resident_slots_by_stride: dict[int, list[list[str]]] = defaultdict(list)
            for entry_name, (tensor_index, _, _) in native_cache_views.items():
                block_stride = native_cache_tensors[tensor_index].block_stride
                if entry_name in reusable_entry_names:
                    reusable_entries_by_stride[block_stride].append(entry_name)
                    continue
                # Different groups own disjoint block IDs in the device pool.
                # Give each resident of the same group a separate fixed slot.
                group_id = self.group_id_by_entry[entry_name]
                resident_slots = resident_slots_by_stride[block_stride]
                for slot_entry_names in resident_slots:
                    if all(
                        self.group_id_by_entry[assigned_entry_name] != group_id
                        for assigned_entry_name in slot_entry_names
                    ):
                        slot_entry_names.append(entry_name)
                        break
                else:
                    resident_slots.append([entry_name])
            for block_stride, resident_slots in resident_slots_by_stride.items():
                for slot_entry_names in resident_slots:
                    slot_alignment = lcm(
                        *(cache_view_alignment(self.spec_by_entry[entry_name]) for entry_name in slot_entry_names)
                    )
                    slot_offset = round_up(backing_bytes, slot_alignment)
                    slot_layouts.append((slot_offset, block_stride, slot_entry_names))
                    slot_bytes = max(cache_view_sizes[entry_name] for entry_name in slot_entry_names)
                    backing_bytes = slot_offset + slot_bytes
        save_dependencies_by_layer = defaultdict(set)
        rotated_entry_names: set[str] = set()
        for block_stride, candidate_entry_names in reusable_entries_by_stride.items():
            # Each pack holds at most one entry per cache group.
            # Start a new pack if the next entry's group is already in the current pack.
            # Cache group ID -> (global layer ID, cache entry name).
            pack: dict[int, tuple[int, str]] = {}
            # Completed packs for this block stride, in the order they were built.
            packs: list[dict[int, tuple[int, str]]] = []
            for entry_name in sorted(
                candidate_entry_names,
                key=lambda entry_name: (
                    self.layer_id_by_entry[entry_name],
                    self.group_id_by_entry[entry_name],
                    entry_name,
                ),
            ):
                group_id = self.group_id_by_entry[entry_name]
                if group_id in pack:
                    packs.append(pack)
                    pack = {}
                pack[group_id] = (self.layer_id_by_entry[entry_name], entry_name)
            if pack:
                packs.append(pack)
            slot_alignment = lcm(
                *(cache_view_alignment(self.spec_by_entry[entry_name]) for entry_name in candidate_entry_names)
            )
            slot_bytes = num_blocks * block_stride
            # Rotation slot index -> cache entry names assigned to that slot.
            entries_by_slot: dict[int, list[str]] = {}
            # Rotation slot index -> last entry assigned per cache group.
            # The inner dict maps group IDs to (global layer ID, cache entry name).
            slot_last_entries: dict[int, dict[int, tuple[int, str]]] = {}
            next_slot = 0
            slot_count = num_shared_buffers
            for pack in packs:
                for distance in range(slot_count):
                    slot_index = (next_slot + distance) % slot_count
                    last_entry_by_group = slot_last_entries.get(slot_index, {})
                    # Keep entries from the same layer and group in separate slots.
                    # In one slot, both would write KV to the same bytes for a block ID.
                    if not any(
                        layer == last_entry_by_group[group_id][0]
                        for group_id, (layer, _) in pack.items()
                        if group_id in last_entry_by_group
                    ):
                        break
                else:
                    slot_index = slot_count
                    slot_count += 1
                rotation_slot_entries = entries_by_slot.get(slot_index)
                if rotation_slot_entries is None:
                    slot_offset = round_up(backing_bytes, slot_alignment)
                    rotation_slot_entries = entries_by_slot[slot_index] = []
                    slot_layouts.append((slot_offset, block_stride, rotation_slot_entries))
                    backing_bytes = slot_offset + slot_bytes
                last_entry_by_group = slot_last_entries.setdefault(slot_index, {})
                for group_id, (layer, entry_name) in pack.items():
                    rotation_slot_entries.append(entry_name)
                    previous_entry = last_entry_by_group.get(group_id)
                    if previous_entry is not None:
                        # This group already used the slot. The current layer must wait
                        # for the previous layer to save its KV before overwriting those bytes.
                        source_layer, source_entry_name = previous_entry
                        save_dependencies_by_layer[layer].add(source_layer)
                        rotated_entry_names.update((source_entry_name, entry_name))
                    last_entry_by_group[group_id] = (layer, entry_name)
                next_slot = (slot_index + 1) % slot_count
        kv_cache_tensors: tuple[KVCacheTensor, ...] = ()
        if save_dependencies_by_layer:
            kv_cache_tensors = tuple(
                KVCacheTensor(
                    size=backing_bytes, layers=entry_names, layer_stride=0, block_stride=block_stride, offset=offset
                )
                for offset, block_stride, entry_names in slot_layouts
            )
        return LayerwiseReuseLayout(
            layers=self.target_layer_ids,
            cache_layer_indices=self.layer_id_by_entry,
            kv_cache_tensors=kv_cache_tensors,
            cache_view_sizes=cache_view_sizes,
            prefetch_layer_map={layer: tuple(sorted(sources)) for layer, sources in save_dependencies_by_layer.items()},
            independent_layers=independent,
            num_prefetch_layers=settings.num_prefetch_layers,
            reused_cache_names=frozenset(rotated_entry_names),
            draft_entry_names=frozenset(self.draft_names),
        )


def plan_layerwise_cache_reuse(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> LayerwiseReuseLayout | None:
    """Build the KV buffer layout for prefill offload on one worker rank."""
    extra_config = get_layerwise_offload_config(getattr(vllm_config, "kv_transfer_config", None))
    if extra_config is None:
        return None
    # An empty device pool has no byte views to reuse.
    if not kv_cache_config.kv_cache_tensors or kv_cache_config.num_blocks == 0 or not kv_cache_config.kv_cache_groups:
        return None
    rules = get_spec_reuse_rules(vllm_config)
    if rules is None:
        return None
    planner = LayerwiseCachePlanner(
        kv_cache_config,
        vllm_config,
        extra_config,
        spec_reuse_rules=rules,
    )
    layout = planner.build()
    if not layout.has_layer_reuse:
        return None

    for name in layout.reused_cache_names:
        spec = planner.spec_by_entry[name]
        if isinstance(spec, MambaSpec) and (
            spec.mamba_cache_mode != "align" or spec.num_prefill_checkpoint_blocks != 0
        ):
            raise NotImplementedError(f"Reusable cache {name} has unsupported state semantics: {type(spec).__name__}")
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        raise ValueError("Layerwise offload does not support num_gpu_blocks_override; use automatic block sizing.")
    if any(tensor.host_resident for tensor in kv_cache_config.kv_cache_tensors) or any(
        getattr(spec, "store_on_host", False) for spec in planner.spec_by_entry.values()
    ):
        raise ValueError("Layerwise offload requires device-resident cache tensors.")
    return layout
