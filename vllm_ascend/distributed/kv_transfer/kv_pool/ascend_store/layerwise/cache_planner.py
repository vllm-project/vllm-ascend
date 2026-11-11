# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select model rules and plan layerwise cache reuse."""

from collections import defaultdict
from math import gcd, lcm
from types import MappingProxyType
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, KVCacheTensor, MambaSpec

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import get_backend_class

from .layerwise_cache_layout import (
    LayerwiseCacheConfig,
    LayerwiseReuseLayout,
    SpecReuseRules,
    get_cache_views,
    get_layerwise_kv_cache_specs,
    get_physical_layer_index,
)

# Resolve model rules only when planning; layout consumers do not load models.
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


def get_layerwise_reuse_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the extra config of the layerwise-reuse connector, if any.

    Layerwise cache reuse requires the GVA backend and an enabled connector.
    """
    if kv_transfer_config is None:
        return None

    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    root_extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    if connector_name in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
        connector_configs = [
            {
                "kv_connector": connector_name,
                "kv_connector_extra_config": root_extra_config,
            }
        ]
    elif connector_name == "MultiConnector":
        connector_configs = root_extra_config.get("connectors", [])
    else:
        return None

    for connector_config in connector_configs:
        if not isinstance(connector_config, dict):
            continue
        if connector_config.get("kv_connector") not in (
            "AscendStoreConnector",
            "MooncakeConnectorStoreV1",
        ):
            continue
        extra_config = connector_config.get("kv_connector_extra_config") or {}
        backend_class = get_backend_class(str(extra_config.get("backend", "mooncake")))
        if backend_class is None or backend_class.layerwise_data_plane != "gva":
            continue
        if extra_config.get("use_layerwise", False):
            return extra_config
    return None


class LayerwiseCachePlanner:
    @staticmethod
    def can_reuse_spec(spec: KVCacheSpec, rules: SpecReuseRules) -> bool:
        """Apply the nearest declared spec rule."""
        for spec_type in type(spec).__mro__:
            if spec_type not in rules:
                continue
            rule = rules[spec_type]
            result = rule(spec) if callable(rule) else rule
            if not isinstance(result, bool):
                raise TypeError("Spec reuse rules must return bool")
            return result
        raise ValueError(f"No reuse rule declared for cache spec {type(spec).__name__}")

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
        self.excluded_layer_names = frozenset(excluded_layer_names or ()) & self.specs.keys()
        self.group_ids = {
            name: index for index, group in enumerate(config.kv_cache_groups) for name in group.layer_names
        }
        base_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.layer_indices = {name: get_physical_layer_index(name, base_layers) for name in self.specs}
        registered = set(getattr(config, "dspark_draft_layer_names", ()))
        self.draft_names = {
            name
            for name, spec in self.specs.items()
            if name in registered or getattr(spec, "is_draft", False) or self.layer_indices[name] >= base_layers
        }

    def _alignment(self, packs: list[tuple[str, ...]], views: dict[str, tuple[int, int, int]]) -> tuple[int, int]:
        strides, dtypes = [], []
        for pack in packs:
            for name in pack:
                tensor = self.config.kv_cache_tensors[views[name][0]]
                strides.append(gcd(tensor.block_stride, tensor.layer_stride))
                spec = self.specs[name]
                dtypes.extend(getattr(spec, "dtypes", (getattr(spec, "dtype", torch.uint8),)))
                if getattr(spec, "scale_dim", 0):
                    dtypes.append(spec.scale_dtype)
        dtype_alignment = lcm(*(get_dtype_size(dtype) for dtype in dtypes))
        return lcm(*strides, dtype_alignment), dtype_alignment

    def build(self) -> LayerwiseReuseLayout:
        indices, draft = self.layer_indices, self.draft_names
        layers = sorted({index for name, index in indices.items() if name not in draft})
        settings = LayerwiseCacheConfig.parse(len(layers), self.extra_config)
        independent = [layers[index] for index in settings.independent_layers]
        tensors = self.config.kv_cache_tensors
        buckets, aliases = defaultdict(list), defaultdict(list)
        regions = {}
        members = set()
        views = {}
        # 1. Bucket layer names by block stride, independent of native slots.
        if settings.has_layer_reuse:
            members = {
                name
                for name, spec in self.specs.items()
                if name not in draft
                and name not in self.excluded_layer_names
                and indices[name] not in independent
                and is_prefix_cacheable(spec)
                and self.can_reuse_spec(spec, self.spec_reuse_rules)
            }
            views = get_cache_views(self.config, self.specs)
            backings = defaultdict(list)
            for name, (index, _, _) in views.items():
                tensor = tensors[index]
                buckets[tensor.block_stride].append(name)
                backings[index].append(name)
            # Retain relative offsets only within genuinely overlapping views.
            # Disjoint strided positions can be placed at a new pack origin.
            for names in backings.values():
                units, unit, end = [], [], -1
                for name in sorted(names, key=lambda name: views[name][1:]):
                    if unit and views[name][1] >= end:
                        units.append(unit)
                        unit = []
                    unit.append(name)
                    end = max(end, views[name][2])
                units.append(unit)
                for unit in units:
                    start = views[unit[0]][1]
                    stride = tensors[views[unit[0]][0]].block_stride
                    size = max(views[name][2] for name in unit) - start
                    aliases[stride].append(unit)
                    for name in unit:
                        regions[name] = (views[name][1] - start, size)
        # Buckets occupy disjoint regions in one device backing. Pack aliases
        # remain intact; resident packs reserve private bytes in that backing.
        candidate = [tensor for tensor in tensors if tensor.host_resident]
        dependencies = defaultdict(set)
        reused_names = set()
        capacity = 0
        for stride, names in buckets.items():
            # Resident aliases stay in their native packs. In particular, a
            # draft or excluded view must not pin unrelated target layers.
            for unit in aliases[stride]:
                if (
                    not all(name in members for name in unit)
                    or max(regions[name][1] for name in unit) > self.config.num_blocks * stride
                ):
                    members.difference_update(unit)
            # 2. Greedily take the longest layer-ordered run without a repeated
            # group. A repeated group starts the next pack, including at the
            # same physical layer; groups are never inferred from spec types.
            packs, pack, groups = [], [], set()
            for name in sorted(names, key=lambda name: (indices[name], self.group_ids[name], name)):
                if name not in members:
                    continue
                group = self.group_ids[name]
                if group in groups:
                    packs.append(tuple(pack))
                    pack, groups = [], set()
                pack.append(name)
                groups.add(group)
            if pack:
                packs.append(tuple(pack))
            packs.extend(tuple(unit) for unit in aliases[stride] if not any(name in members for name in unit))
            packs.sort(key=lambda pack: min(indices[name] for name in pack))
            pack_ids = {name: index for index, pack in enumerate(packs) for name in pack}
            if any(len({pack_ids[name] for name in unit}) != 1 for unit in aliases[stride]):
                # Keep this bucket resident rather than split a native alias;
                # other stride buckets can still use their own shared packs.
                packs = [tuple(unit) for unit in aliases[stride]]
                members.difference_update(names)
            descriptors, cursor = {}, 0
            alignment, dtype_alignment = self._alignment(packs, views)
            # Align in per-block units so probing one block gives exactly the
            # same cost as planning the final number of blocks.
            unit = self.config.num_blocks * dtype_alignment
            capacity = (capacity + unit - 1) // unit * unit
            buffer_size = self.config.num_blocks * stride
            buffers, previous, slot = {}, {}, 0
            for pack in packs:
                # 3. Rotate complete packs within this bucket. Resident packs
                # reserve private regions and do not advance the rotation.
                group_layers = {(self.group_ids[name], indices[name]) for name in pack}
                size = max(regions[name][1] for name in pack)
                reusable = (
                    all(name in members for name in pack)
                    and size <= buffer_size
                    and all(layer > previous.get((slot, group), (-1, ()))[0] for group, layer in group_layers)
                )
                if reusable and slot in buffers:
                    offset = buffers[slot]
                else:
                    offset = (cursor + alignment - 1) // alignment * alignment
                    cursor = offset + (buffer_size if reusable else size)
                    if reusable:
                        buffers[slot] = offset
                group_names = defaultdict(set)
                for name in pack:
                    block_stride = tensors[views[name][0]].block_stride
                    cache_offset = offset + regions[name][0]
                    key = (cache_offset, block_stride)
                    if key not in descriptors:
                        descriptors[key] = KVCacheTensor(
                            size=0,
                            layers=[],
                            layer_stride=0,
                            block_stride=block_stride,
                            offset=capacity + cache_offset,
                        )
                    descriptors[key].layers.append(name)
                    group_names[self.group_ids[name]].add(name)
                if reusable:
                    for group, layer in group_layers:
                        key = (slot, group)
                        if key in previous:
                            source, names = previous[key]
                            dependencies[layer].add(source)
                            reused_names.update(names | group_names[group])
                        previous[key] = (layer, group_names[group])
                    slot = (slot + 1) % settings.num_shared_buffers
            capacity += cursor
            candidate.extend(descriptors.values())
        for tensor in candidate:
            if not tensor.host_resident:
                tensor.size = capacity
        if not dependencies:
            candidate = tensors
            dependencies.clear()
            reused_names.clear()
        return LayerwiseReuseLayout(
            layers=tuple(layers),
            cache_layer_indices=indices,
            kv_cache_tensors=tuple(candidate),
            cache_view_sizes={name: end - start for name, (_, start, end) in views.items()},
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
            if self.vllm_config.parallel_config.prefill_context_parallel_size > 1:
                raise NotImplementedError("Layerwise offload does not support prefill context parallelism.")
            logger.info(
                "Layerwise reuse uses one %d-byte device backing; independent layers=%s.",
                next(tensor.size for tensor in layout.kv_cache_tensors if not tensor.host_resident),
                layout.independent_layers,
            )
            self.config.kv_cache_tensors = list(layout.kv_cache_tensors)
            self.config.layerwise_kv_cache_reuse = True
        self.config.layerwise_reuse_layout = layout
        if self.draft_names:
            self.config.layerwise_draft_layer_names = tuple(sorted(self.draft_names))


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    *,
    excluded_layer_names: set[str] | None = None,
) -> None:
    """Plan whole-pack reuse using registered spec rules; keep cache identity."""
    layout = getattr(kv_cache_config, "layerwise_reuse_layout", None)
    if layout is not None:
        if excluded_layer_names and layout.reused_cache_names.intersection(excluded_layer_names):
            raise ValueError("Reuse exclusions must be supplied before the cache layout is planned.")
        return
    extra_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    # An empty device pool has no byte views to reuse.
    if extra_config is None or not kv_cache_config.kv_cache_tensors or kv_cache_config.num_blocks == 0:
        return
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    if not LayerwiseCacheConfig.has_layer_reuse_for_stage(base_layers, extra_config):
        return
    rules = get_spec_reuse_rules(vllm_config)
    if rules is None:
        raise NotImplementedError("Layerwise offload requires registered model spec reuse rules.")
    LayerwiseCachePlanner(
        kv_cache_config, vllm_config, extra_config, excluded_layer_names, spec_reuse_rules=rules
    ).apply()
