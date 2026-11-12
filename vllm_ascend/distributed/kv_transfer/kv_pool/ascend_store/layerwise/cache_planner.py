# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan layerwise cache reuse from native cache groups."""

from collections import defaultdict
from collections.abc import Iterator
from math import lcm
from types import MappingProxyType
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
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

_STORE_CONNECTORS = ("AscendStoreConnector", "MooncakeConnectorStoreV1")
SPEC_REUSE_RULES = MappingProxyType(
    dict.fromkeys(
        ("glm5_next", "glm5_next_text"),
        "vllm_ascend.models.glm5next.cache_reuse.GLM5_NEXT_SPEC_REUSE_RULES",
    )
)


def get_spec_reuse_rules(vllm_config: VllmConfig) -> SpecReuseRules | None:
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    path = SPEC_REUSE_RULES.get(getattr(text_config, "model_type", None))
    return resolve_obj_by_qualname(path) if path is not None else None


def _iter_layerwise_store_configs(kv_transfer_config: Any) -> Iterator[dict[str, Any]]:
    if kv_transfer_config is None:
        return

    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    root_extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    if connector_name in _STORE_CONNECTORS:
        connector_configs = [
            {
                "kv_connector": connector_name,
                "kv_connector_extra_config": root_extra_config,
            }
        ]
    elif connector_name == "MultiConnector":
        connector_configs = root_extra_config.get("connectors", [])
    else:
        return

    for connector_config in connector_configs:
        if not isinstance(connector_config, dict):
            continue
        if connector_config.get("kv_connector") not in _STORE_CONNECTORS:
            continue
        extra_config = connector_config.get("kv_connector_extra_config") or {}
        if extra_config.get("use_layerwise", False):
            yield extra_config


def _uses_gva_backend(extra_config: dict[str, Any]) -> bool:
    backend_class = get_backend_class(str(extra_config.get("backend", "mooncake")))
    return backend_class is not None and backend_class.layerwise_data_plane == "gva"


def get_layerwise_gva_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the extra config for a layerwise GVA store."""
    for extra_config in _iter_layerwise_store_configs(kv_transfer_config):
        if _uses_gva_backend(extra_config):
            return extra_config
    return None


def get_layerwise_offload_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Find a layerwise GVA store with an explicit shared-buffer count."""
    for extra_config in _iter_layerwise_store_configs(kv_transfer_config):
        if extra_config.get("layerwise_num_shared_buffers") is not None and _uses_gva_backend(extra_config):
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
        excluded_layer_names: set[str] | None = None,
        *,
        spec_reuse_rules: SpecReuseRules,
    ) -> None:
        self.config = config
        self.vllm_config = vllm_config
        self.extra_config = extra_config
        self.spec_reuse_rules = spec_reuse_rules
        self.specs = get_layerwise_kv_cache_specs(config)
        self.group_ids = {
            name: index for index, group in enumerate(config.kv_cache_groups) for name in group.layer_names
        }
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.layer_indices = {name: get_physical_layer_index(name, self.total_layers) for name in self.specs}
        dspark_draft_names = set(getattr(config, "dspark_draft_layer_names", ()))
        self.draft_names = {
            name
            for name, spec in self.specs.items()
            if name in dspark_draft_names
            or getattr(spec, "is_draft", False)
            or self.layer_indices[name] >= self.total_layers
        }
        self.layers = tuple(
            sorted({index for name, index in self.layer_indices.items() if name not in self.draft_names})
        )
        self.excluded_layer_names = self.draft_names | (frozenset(excluded_layer_names or ()) & self.specs.keys())

    def build(self) -> LayerwiseReuseLayout:
        indices, layers = self.layer_indices, self.layers
        settings = LayerwiseCacheConfig.parse(self.total_layers, self.extra_config, stage_layers=layers)
        independent = settings.independent_layers
        num_shared_buffers = settings.num_shared_buffers or 0
        native_cache_tensors = self.config.kv_cache_tensors
        num_blocks = self.config.num_blocks
        reusable_entries_by_stride: dict[int, list[str]] = defaultdict(list)
        view_bytes_by_entry: dict[str, int] = {}
        slot_layouts: list[tuple[int, int, list[str]]] = []
        backing_bytes = 0
        reusable_entry_names = set()
        if num_shared_buffers:
            reusable_entry_names = {
                entry_name
                for group in self.config.kv_cache_groups
                if is_prefix_cacheable(group.kv_cache_spec)
                for entry_name in group.layer_names
                if entry_name not in self.excluded_layer_names
                and indices[entry_name] not in independent
                and self.can_reuse_spec(self.specs[entry_name], self.spec_reuse_rules)
            }
        if reusable_entry_names:
            native_cache_views = get_cache_views(self.config, self.specs)
            view_bytes_by_entry = {
                entry_name: view_end - view_start
                for entry_name, (_, view_start, view_end) in native_cache_views.items()
            }
            resident_slots_by_stride: dict[int, list[list[str]]] = defaultdict(list)
            for entry_name, (tensor_index, _, _) in native_cache_views.items():
                block_stride = native_cache_tensors[tensor_index].block_stride
                if entry_name in reusable_entry_names:
                    reusable_entries_by_stride[block_stride].append(entry_name)
                    continue
                # Different groups own disjoint block IDs in the device pool.
                # Give each resident of the same group a separate fixed slot.
                group_id = self.group_ids[entry_name]
                slots_for_stride = resident_slots_by_stride[block_stride]
                for slot_entry_names in slots_for_stride:
                    if all(self.group_ids[member_name] != group_id for member_name in slot_entry_names):
                        slot_entry_names.append(entry_name)
                        break
                else:
                    slots_for_stride.append([entry_name])
            for block_stride, slots_for_stride in resident_slots_by_stride.items():
                for slot_entry_names in slots_for_stride:
                    dtype_alignment = lcm(
                        *(cache_view_alignment(self.specs[entry_name]) for entry_name in slot_entry_names)
                    )
                    slot_offset = round_up(backing_bytes, dtype_alignment)
                    slot_layouts.append((slot_offset, block_stride, slot_entry_names))
                    slot_size = max(view_bytes_by_entry[entry_name] for entry_name in slot_entry_names)
                    backing_bytes = slot_offset + slot_size
        dependencies = defaultdict(set)
        reused_names = set()
        for block_stride, bucket_entries in reusable_entries_by_stride.items():
            # A repeated group starts a new pack, including at the same layer.
            pack: dict[int, tuple[int, str]] = {}
            packs: list[dict[int, tuple[int, str]]] = []
            for entry_name in sorted(
                bucket_entries, key=lambda entry_name: (indices[entry_name], self.group_ids[entry_name], entry_name)
            ):
                group_id = self.group_ids[entry_name]
                if group_id in pack:
                    packs.append(pack)
                    pack = {}
                pack[group_id] = (indices[entry_name], entry_name)
            if pack:
                packs.append(pack)
            dtype_alignment = lcm(*(cache_view_alignment(self.specs[entry_name]) for entry_name in bucket_entries))
            buffer_size = num_blocks * block_stride
            entries_by_slot: dict[int, list[str]] = {}
            slot_last_entries: dict[int, dict[int, tuple[int, str]]] = {}
            next_slot = 0
            slot_count = num_shared_buffers
            for pack in packs:
                for distance in range(slot_count):
                    slot_index = (next_slot + distance) % slot_count
                    last_entry_by_group = slot_last_entries.get(slot_index, {})
                    # Same-layer entries in a group use the same block addresses.
                    if not any(
                        layer == last_entry_by_group[group_id][0]
                        for group_id, (layer, _) in pack.items()
                        if group_id in last_entry_by_group
                    ):
                        break
                else:
                    slot_index = slot_count
                    slot_count += 1
                slot_entry_names = entries_by_slot.get(slot_index)
                if slot_entry_names is None:
                    slot_offset = round_up(backing_bytes, dtype_alignment)
                    slot_entry_names = entries_by_slot[slot_index] = []
                    slot_layouts.append((slot_offset, block_stride, slot_entry_names))
                    backing_bytes = slot_offset + buffer_size
                last_entry_by_group = slot_last_entries.setdefault(slot_index, {})
                for group_id, (layer, entry_name) in pack.items():
                    slot_entry_names.append(entry_name)
                    previous_entry = last_entry_by_group.get(group_id)
                    if previous_entry is not None:
                        source_layer, source_entry_name = previous_entry
                        dependencies[layer].add(source_layer)
                        reused_names.update((source_entry_name, entry_name))
                    last_entry_by_group[group_id] = (layer, entry_name)
                next_slot = (slot_index + 1) % slot_count
        kv_cache_tensors: tuple[KVCacheTensor, ...] = ()
        if dependencies:
            kv_cache_tensors = tuple(
                KVCacheTensor(
                    size=backing_bytes, layers=entry_names, layer_stride=0, block_stride=block_stride, offset=offset
                )
                for offset, block_stride, entry_names in slot_layouts
            )
        return LayerwiseReuseLayout(
            layers=layers,
            cache_layer_indices=indices,
            kv_cache_tensors=kv_cache_tensors,
            cache_view_sizes=view_bytes_by_entry,
            prefetch_layer_map={layer: tuple(sorted(sources)) for layer, sources in dependencies.items()},
            independent_layers=independent,
            num_prefetch_layers=settings.num_prefetch_layers,
            reused_cache_names=frozenset(reused_names),
        )

    def apply(self) -> None:
        layout = self.build()
        for name in layout.reused_cache_names:
            spec = self.specs[name]
            if isinstance(spec, MambaSpec) and (
                spec.mamba_cache_mode != "align" or getattr(spec, "num_prefill_checkpoint_blocks", 0) != 0
            ):
                raise NotImplementedError(
                    f"Reusable cache {name} has unsupported state semantics: {type(spec).__name__}"
                )
        if layout.has_layer_reuse:
            if getattr(getattr(self.vllm_config, "cache_config", None), "num_gpu_blocks_override", None) is not None:
                raise ValueError(
                    "Layerwise offload does not support num_gpu_blocks_override; use automatic block sizing."
                )
            if any(tensor.host_resident for tensor in self.config.kv_cache_tensors) or any(
                getattr(spec, "store_on_host", False) for spec in self.specs.values()
            ):
                raise ValueError("Layerwise offload requires device-resident cache tensors.")
            logger.info(
                "Layerwise reuse uses one %d-byte device backing; independent layers=%s.",
                next(tensor.size for tensor in layout.kv_cache_tensors if not tensor.host_resident),
                layout.independent_layers,
            )
            self.config.kv_cache_tensors = list(layout.kv_cache_tensors)
            self.config.layerwise_kv_cache_reuse = True
        self.config.layerwise_reuse_layout = layout
        if layout.has_layer_reuse and self.draft_names:
            self.config.layerwise_draft_layer_names = tuple(sorted(self.draft_names))


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    *,
    excluded_layer_names: set[str] | None = None,
) -> None:
    """Apply whole-pack reuse when the model and stage request it."""
    layout = getattr(kv_cache_config, "layerwise_reuse_layout", None)
    if layout is not None:
        if excluded_layer_names and layout.reused_cache_names.intersection(excluded_layer_names):
            raise ValueError("Reuse exclusions must be supplied before the cache layout is planned.")
        return
    # An empty device pool has no byte views to reuse.
    if not kv_cache_config.kv_cache_tensors or kv_cache_config.num_blocks == 0 or not kv_cache_config.kv_cache_groups:
        return
    extra_config = get_layerwise_offload_config(getattr(vllm_config, "kv_transfer_config", None))
    if extra_config is None:
        return
    rules = get_spec_reuse_rules(vllm_config)
    if rules is None:
        return
    model_config = vllm_config.model_config
    planner = LayerwiseCachePlanner(
        kv_cache_config,
        vllm_config,
        extra_config,
        excluded_layer_names,
        spec_reuse_rules=rules,
    )
    parallel_config = vllm_config.parallel_config
    pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
    if pp_size > 1:
        # Core plans every worker's groups with the same parallel config.
        complete_stage = any(
            planner.layers == tuple(range(*get_pp_indices(planner.total_layers, rank, pp_size)))
            for rank in range(pp_size)
        )
    else:
        complete_stage = len(planner.layers) >= model_config.get_num_layers(parallel_config)
    if not complete_stage:
        # The hook cursor requires the complete local set of cache-bearing layers.
        planner.extra_config = {**extra_config, "layerwise_independent_layers": "all"}
    planner.apply()
