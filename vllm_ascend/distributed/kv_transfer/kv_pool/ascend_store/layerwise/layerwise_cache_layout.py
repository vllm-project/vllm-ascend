from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.core.layerwise_cache_config import (
    LayerwisePolicy,
    LayerwiseReuseLayout,
    clear_cached_group_properties,
    get_layerwise_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_backend_class,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers


def allocate_shared_indexer_cache(
    specs: dict[str, AscendSFAIndexerCacheSpec],
    num_blocks: int,
    allocate: Callable[[int], torch.Tensor],
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Share raw bytes while retaining each layer's data and scale layout.

    The caller must fence reuse between layers. Returned byte views contain
    exactly the layer's logical capacity, so reshape and transfer code never
    mistake spare backing capacity for extra blocks.
    """
    regions: dict[str, tuple[int, int, int]] = {}
    capacity = 0
    for name, spec in specs.items():
        rows = num_blocks * spec.sfa_dcp_replicated_indexer_size * spec.block_size * spec.num_kv_heads
        data_bytes = rows * spec.head_size * get_dtype_size(spec.dtype)
        scale_bytes = rows * spec.scale_dim * get_dtype_size(spec.scale_dtype)
        scale_alignment = get_dtype_size(spec.scale_dtype)
        scale_offset = ((data_bytes + scale_alignment - 1) // scale_alignment) * scale_alignment
        regions[name] = (data_bytes, scale_offset, scale_bytes)
        capacity = max(capacity, scale_offset + scale_bytes if scale_bytes else data_bytes)
    backing = allocate(capacity)
    views: dict[str, tuple[torch.Tensor, ...]] = {}
    region_views: dict[tuple[int, int, int], tuple[torch.Tensor, ...]] = {}
    for name, region in regions.items():
        if region not in region_views:
            data_bytes, scale_offset, scale_bytes = region
            data = backing[:data_bytes]
            region_views[region] = (
                (data, backing[scale_offset : scale_offset + scale_bytes]) if scale_bytes else (data,)
            )
        views[name] = region_views[region]
    return views


def get_physical_layer_index(layer_name: str, base_layers: int) -> int:
    match = re.search(
        r"(?:^|\.)mtp(?:\.layers)?\.(\d+)(?:\.|$)",
        layer_name,
    )
    if match:
        return base_layers + int(match.group(1))
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", layer_name)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class LayerwiseCacheLayout:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    prefetch_layer_map: dict[int, int]
    storage_indices: list[list[int]]
    has_layer_reuse: bool


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


DEFAULT_MAX_PREFETCH_LAYERS = 8


def parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def normalize_layerwise_policy(
    physical_layers: list[int] | range,
    extra_config: dict[str, Any] | None = None,
    excluded_owners: tuple[str, ...] = (),
) -> LayerwisePolicy:
    layers = sorted(set(physical_layers))
    if not layers:
        raise ValueError("num_layers must be at least 1")
    config = extra_config or {}
    shared = config.get("layerwise_num_shared_buffers")
    shared = len(layers) if shared is None else parse_int_config(shared, "layerwise_num_shared_buffers")
    prefetch = config.get("layerwise_prefetch_layers")
    prefetch = (
        min(shared, DEFAULT_MAX_PREFETCH_LAYERS)
        if prefetch is None
        else parse_int_config(prefetch, "layerwise_prefetch_layers")
    )
    if shared < 1 or prefetch < 1:
        raise ValueError("layerwise shared buffers and prefetch layers must be at least 1")
    independent = config.get("layerwise_independent_layers")
    if independent is None:
        indices = [0]
    elif isinstance(independent, str) and independent.strip().lower() == "all":
        indices = list(range(len(layers)))
    elif isinstance(independent, list):
        indices = [parse_int_config(index, "layerwise_independent_layers") for index in independent]
    else:
        raise TypeError("layerwise_independent_layers must be a list of integers or 'all'")
    normalized = set()
    for index in indices:
        index = index + len(layers) if index < 0 else index
        if not 0 <= index < len(layers):
            raise ValueError(f"layerwise_independent_layers contains out-of-range layer index {index}")
        normalized.add(layers[index])
    return LayerwisePolicy(shared, prefetch, tuple(sorted(normalized)), tuple(sorted(set(excluded_owners))))


def build_layerwise_cache_layout(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseCacheLayout:
    policy = normalize_layerwise_policy(range(num_layers), extra_config)
    num_shared_buffers = policy.num_shared_buffers
    num_prefetch_layers = policy.num_prefetch_layers
    independent_layers = list(policy.independent_layers)

    independent_layer_set = set(independent_layers)
    reused_layers = [index for index in range(num_layers) if index not in independent_layer_set]
    has_layer_reuse = len(reused_layers) > num_shared_buffers
    prefetch_layer_map = {
        reused_layers[next_index]: reused_layers[next_index - num_shared_buffers]
        for next_index in range(num_shared_buffers, len(reused_layers))
    }
    storage_indices = [[layer] for layer in independent_layers]
    for slot in range(num_shared_buffers):
        members = list(range(slot, len(reused_layers), num_shared_buffers))
        if members:
            storage_indices.append([reused_layers[index] for index in members])

    return LayerwiseCacheLayout(
        num_shared_buffers=num_shared_buffers,
        num_prefetch_layers=num_prefetch_layers,
        independent_layers=independent_layers,
        prefetch_layer_map=prefetch_layer_map,
        storage_indices=storage_indices,
        has_layer_reuse=has_layer_reuse,
    )


def get_layerwise_kv_cache_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    """Expand specs without losing per-name intent after homogeneous merging."""
    result = {}
    for group in kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            spec = (
                group.kv_cache_spec.kv_cache_specs[name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
            result[name] = spec
    return result


def _storage_domains(config, specs):
    """Validate affine regions and resolve explicit backing identity.

    An unmarked descriptor is an independent allocation. Shared-storage
    adapters must declare backing_id and the physical block stride; the
    planner neither recognizes model names nor guesses ownership from size.
    """
    domains = {}
    covered = set()
    for index, tensor in enumerate(config.kv_cache_tensors):
        names = get_kv_cache_tensor_layers(tensor)
        shared = tensor.backing_id is not None
        key = ("shared", tensor.backing_id) if shared else ("private", index)
        domains.setdefault(key, []).append(tensor)
        for ordinal, name in enumerate(names):
            if name in covered or name not in specs:
                raise ValueError(f"Duplicate or unknown layerwise descriptor: {name}")
            covered.add(name)
            spec = specs[name]
            if not spec.reuse and not shared:
                continue
            if spec.reuse and (
                not isinstance(spec, (AttentionSpec, MambaSpec))
                or (isinstance(spec, MambaSpec) and not shared)
                or getattr(spec, "model_version", None)
            ):
                raise NotImplementedError(f"Layerwise save/restore is unsupported for {name}: {type(spec).__name__}")
            page = getattr(spec, "real_page_size_bytes", spec.page_size_bytes) if shared else spec.page_size_bytes
            stride = tensor.block_stride
            if stride < page or (not shared and stride != page):
                raise NotImplementedError(f"Cannot resolve independent contiguous regions for {name}")
            if len(names) > 1 and tensor.layer_stride < config.num_blocks * stride:
                if not (get_layerwise_layout(config) is not None and tensor.layer_stride == 0):
                    raise NotImplementedError(f"Cannot resolve independent contiguous regions for {name}")
            start = tensor.offset + ordinal * tensor.layer_stride
            if start < 0 or stride <= 0 or start + config.num_blocks * stride > tensor.size:
                raise ValueError(f"Layerwise descriptor exceeds backing: {name}")
    if covered != set(specs):
        raise ValueError("Layerwise descriptors must cover every cache name")
    group_ids = {name: i for i, group in enumerate(config.kv_cache_groups) for name in group.layer_names}
    for (kind, _), tensors in domains.items():
        if kind != "shared":
            continue
        if len({tensor.size for tensor in tensors}) != 1:
            raise ValueError("Descriptors for one backing disagree about capacity")
        regions = _layer_regions(tensors, config.num_blocks)
        names = list(regions)
        for i, name in enumerate(names):
            start, end = regions[name]
            for other in names[:i]:
                other_start, other_end = regions[other]
                if start >= other_end or other_start >= end:
                    continue
                if (start, end) != (other_start, other_end):
                    raise NotImplementedError("Partially overlapping packed regions cannot be replanned safely")
                if group_ids[name] == group_ids[other] and get_layerwise_layout(config) is None:
                    raise ValueError("Original simultaneous cache regions alias within one group")
    return list(domains.values())


def _layer_regions(tensors, num_blocks):
    return {
        name: (
            tensor.offset + i * tensor.layer_stride,
            tensor.offset + i * tensor.layer_stride + num_blocks * tensor.block_stride,
        )
        for tensor in tensors
        for i, name in enumerate(tensor.layers)
    }


def _layer_rotation(layers: list[int], buffers: int) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    """Assign all components of a physical layer to the same logical buffer."""
    if buffers == 1 and len(layers) > 1:
        raise ValueError("Layerwise rotation across multiple layers requires at least two buffers")
    layer_buffers = {layer: index % buffers for index, layer in enumerate(layers)}
    predecessors = {layer: (previous,) for layer, previous in zip(layers[buffers:], layers)}
    return layer_buffers, predecessors


def _descriptor_slots(domains, specs, group_ids, layers, layer_buffers):
    """Choose disjoint byte regions within each logical buffer and backing."""
    result = []
    for tensors in domains:
        by_name = {name: tensor for tensor in tensors for name in tensor.layers if specs[name].reuse}
        buckets = {}
        for name in sorted(by_name, key=lambda name: (layers[name], name)):
            layer = layers[name]
            tensor = by_name[name]
            key = (layer_buffers[layer], tensor.offset, tensor.layer_stride, tensor.block_stride)
            slots = buckets.setdefault(key, [])
            for members, used_layers in slots:
                if layer in used_layers:
                    continue
                if all(_can_rotate_owners(name, other, by_name, specs, group_ids) for other in members):
                    members.append(name)
                    used_layers.add(layer)
                    break
            else:
                slots.append(([name], {layer}))
        for slots in buckets.values():
            result.extend(tuple(members) for members, _ in slots)
    return tuple(result)


def _can_rotate_owners(name, other, tensors, specs, group_ids):
    # The caller already matched backing, original geometry and logical buffer.
    if tensors[name] is tensors[other]:
        return type(specs[name]) is type(specs[other]) and specs[name] == specs[other]
    return tensors[name].backing_id is not None and group_ids[name] != group_ids[other]


def get_layerwise_policy(
    config: KVCacheConfig,
    base_layers: int,
    extra_config: dict[str, Any],
    excluded_layer_names: set[str] | None = None,
) -> LayerwisePolicy:
    layers = sorted({get_physical_layer_index(name, base_layers) for name in get_layerwise_kv_cache_specs(config)})
    return normalize_layerwise_policy(layers, extra_config, tuple(sorted(excluded_layer_names or ())))


def _apply_independent_layers(group: KVCacheGroupSpec, independent: set[int], base_layers: int) -> KVCacheGroupSpec:
    spec = group.kv_cache_spec
    named = isinstance(spec, UniformTypeKVCacheSpecs)
    members = spec.kv_cache_specs if named else dict.fromkeys(group.layer_names, spec)
    resident = [
        name
        for name in group.layer_names
        if members[name].reuse and get_physical_layer_index(name, base_layers) in independent
    ]
    if not resident:
        return group
    if not named and len(resident) == len(group.layer_names):
        return replace(group, kv_cache_spec=replace(spec, reuse=False))
    effective = dict(members)
    for name in resident:
        effective[name] = replace(members[name], reuse=False)
    if named:
        spec = replace(spec, kv_cache_specs=effective)
    else:
        spec = UniformTypeKVCacheSpecs(block_size=spec.block_size, kv_cache_specs=effective)
    return replace(group, kv_cache_spec=spec)


def _validate_reuse_intent(config: KVCacheConfig, policy: LayerwisePolicy, base_layers: int) -> None:
    specs = get_layerwise_kv_cache_specs(config)
    for name, spec in specs.items():
        if spec.reuse and (
            name in policy.excluded_owners or get_physical_layer_index(name, base_layers) in policy.independent_layers
        ):
            raise ValueError(f"reuse=True conflicts with excluded/independent configuration: {name}")
    for group in config.kv_cache_groups:
        if any(specs[name].reuse for name in group.layer_names) and (
            group.host_resident or not group.enable_kv_transfer
        ):
            raise ValueError("Reused cache must support device save/restore")
        units = {}
        for name in group.layer_names:
            if name not in policy.excluded_owners:
                units.setdefault(get_physical_layer_index(name, base_layers), set()).add(specs[name].reuse)
        if any(len(flags) > 1 for flags in units.values()):
            raise NotImplementedError("Mixed reuse intent within one group/layer transfer is unsupported")


def validate_layerwise_reuse_layout(
    config: KVCacheConfig,
    layout: LayerwiseReuseLayout,
    *,
    policy: LayerwisePolicy | None = None,
    base_layers: int = 0,
) -> None:
    """Check a published placement without inferring or changing its slots."""
    policy = layout.policy if policy is None else policy
    _validate_reuse_intent(config, policy, base_layers)
    if (
        replace(layout.policy, excluded_owners=policy.excluded_owners) != policy
        or layout.num_blocks != config.num_blocks
    ):
        raise ValueError("Layerwise policy/capacity changed after placement")
    specs = get_layerwise_kv_cache_specs(config)
    requested = {name for name, spec in specs.items() if spec.reuse}
    members = [name for slot in layout.buffer_slots for name in slot]
    if (
        requested != set(members)
        or len(members) != len(set(members))
        or layout.resident_layer_names != set(specs) - requested
    ):
        raise ValueError("Layerwise reuse intent changed after placement")
    layers = {name: get_physical_layer_index(name, base_layers) for name in requested}
    layer_buffers, predecessors = _layer_rotation(sorted(set(layers.values())), policy.num_shared_buffers)
    if layout.layer_buffers != layer_buffers or layout.predecessors != (predecessors if layout.has_layer_reuse else {}):
        raise ValueError("Layerwise buffer assignment or predecessor changed after placement")
    for slot in layout.buffer_slots:
        slot_layers = {layers[name] for name in slot}
        if len(slot_layers) != len(slot) or len({layer_buffers[layer] for layer in slot_layers}) != 1:
            raise ValueError("Layerwise descriptor spans conflicting buffer assignments")
    if layout.has_layer_reuse:
        try:
            valid = build_layerwise_tensors(config, layout) == config.kv_cache_tensors
        except (ValueError, NotImplementedError):
            valid = False
        if not valid:
            raise ValueError("Layerwise descriptors changed after placement")


def infer_layerwise_reuse_layout(
    kv_cache_config: KVCacheConfig,
    policy: LayerwisePolicy,
    *,
    base_layers: int = 0,
) -> tuple[KVCacheConfig, LayerwiseReuseLayout]:
    """Return a config with effective per-layer specs and its layout, leaving the input untouched."""
    if get_layerwise_layout(kv_cache_config) is not None:
        raise ValueError("Cannot infer layerwise reuse from an already placed configuration")
    independent = set(policy.independent_layers)
    groups = [_apply_independent_layers(group, independent, base_layers) for group in kv_cache_config.kv_cache_groups]
    kv_cache_config = replace(kv_cache_config, kv_cache_groups=groups)
    _validate_reuse_intent(kv_cache_config, policy, base_layers)
    specs = get_layerwise_kv_cache_specs(kv_cache_config)
    if not any(spec.reuse for spec in specs.values()):
        return kv_cache_config, LayerwiseReuseLayout(policy, kv_cache_config.num_blocks, (), frozenset(specs), {}, {})
    if not kv_cache_config.kv_cache_tensors:
        raise ValueError("Requested layerwise reuse has no original cache descriptors")
    domains = _storage_domains(kv_cache_config, specs)
    layers = {name: get_physical_layer_index(name, base_layers) for name, spec in specs.items() if spec.reuse}
    layer_buffers, predecessors = _layer_rotation(sorted(set(layers.values())), policy.num_shared_buffers)
    group_ids = {name: i for i, group in enumerate(kv_cache_config.kv_cache_groups) for name in group.layer_names}
    slots = _descriptor_slots(domains, specs, group_ids, layers, layer_buffers)
    return kv_cache_config, LayerwiseReuseLayout(
        policy,
        kv_cache_config.num_blocks,
        slots,
        frozenset(name for name, spec in specs.items() if not spec.reuse),
        predecessors if any(len(slot) > 1 for slot in slots) else {},
        layer_buffers,
    )


def _pack_shared_domain(config, tensors, layout, specs):
    """Pack resident banks and rotating slots, retaining proven pool aliases."""
    by_name = {name: tensor for tensor in tensors for name in tensor.layers}
    group_ids = {name: i for i, group in enumerate(config.kv_cache_groups) for name in group.layer_names}
    residents = sorted(
        set(by_name) & layout.resident_layer_names,
        key=lambda name: (get_physical_layer_index(name, 0), name),
    )
    slots = [slot for slot in layout.buffer_slots if slot[0] in by_name]
    cursor = 0
    offsets = {}
    # Keep resident banks separate from every rotating bank, even if the
    # original pool overlaid their groups. Component offsets stay unchanged.
    for name in residents:
        offsets[name] = cursor
        cursor += config.num_blocks * by_name[name].block_stride
    for slot in slots:
        offsets.update(dict.fromkeys(slot, cursor))
        cursor += config.num_blocks * by_name[slot[0]].block_stride

    result = []
    # Consecutive resident regions of identical geometry have an ordinary
    # nonzero layer stride, so one descriptor can retain many layer views.
    for name in residents:
        source = by_name[name]
        previous = result[-1] if result else None
        if (
            previous is not None
            and group_ids[previous.layers[0]] == group_ids[name]
            and type(specs[previous.layers[0]]) is type(specs[name])
            and specs[previous.layers[0]] == specs[name]
            and previous.block_stride == source.block_stride
            and previous.offset + len(previous.layers) * previous.layer_stride == offsets[name]
        ):
            previous.layers.append(name)
        else:
            result.append(
                replace(
                    source,
                    layers=[name],
                    size=cursor,
                    offset=offsets[name],
                    layer_stride=config.num_blocks * source.block_stride,
                )
            )
    for slot in slots:
        # Typed views and scheduler block tables are still resolved per owner.
        result.append(
            replace(by_name[slot[0]], layers=list(slot), size=cursor, offset=offsets[slot[0]], layer_stride=0)
        )
    return result


def build_layerwise_tensors(config: KVCacheConfig, layout: LayerwiseReuseLayout) -> list[KVCacheTensor]:
    specs = get_layerwise_kv_cache_specs(config)
    domains = _storage_domains(config, specs)
    slot_by_name = {name: slot for slot in layout.buffer_slots for name in slot}
    result = []
    for tensors in domains:
        if not any(name in slot_by_name for tensor in tensors for name in tensor.layers):
            result.extend(tensors)
            continue
        if tensors[0].backing_id is not None:
            result.extend(_pack_shared_domain(config, tensors, layout, specs))
            continue
        emitted = set()
        for tensor in tensors:
            for name in tensor.layers:
                slot = slot_by_name.get(name, (name,))
                if slot in emitted:
                    continue
                emitted.add(slot)
                page = specs[name].page_size_bytes
                result.append(
                    replace(
                        tensor,
                        layers=list(slot),
                        size=config.num_blocks * page,
                        layer_stride=0,
                        block_stride=page,
                        offset=0,
                    )
                )
    return result


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    *,
    excluded_layer_names: set[str] | None = None,
) -> None:
    """Validate and publish descriptors plus their necessary overwrite fences."""
    extra = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if extra is None:
        return
    if not kv_cache_config.kv_cache_tensors and not any(
        spec.reuse for spec in get_layerwise_kv_cache_specs(kv_cache_config).values()
    ):
        return
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    policy = get_layerwise_policy(kv_cache_config, base_layers, extra, excluded_layer_names)
    previous = get_layerwise_layout(kv_cache_config)
    if previous is not None:
        validate_layerwise_reuse_layout(kv_cache_config, previous, policy=policy, base_layers=base_layers)
        return
    effective, layout = infer_layerwise_reuse_layout(kv_cache_config, policy, base_layers=base_layers)
    tensors = build_layerwise_tensors(effective, layout) if layout.has_layer_reuse else kv_cache_config.kv_cache_tensors
    names = [name for tensor in tensors for name in get_kv_cache_tensor_layers(tensor)]
    if kv_cache_config.kv_cache_tensors and (
        len(names) != len(set(names)) or set(names) != set(get_layerwise_kv_cache_specs(effective))
    ):
        raise ValueError("Layerwise conversion lost or duplicated cache names")
    if any(tensor.layer_stride < 0 or tensor.block_stride <= 0 or tensor.offset < 0 for tensor in tensors):
        raise ValueError("Layerwise conversion produced an invalid descriptor")
    kv_cache_config.kv_cache_groups = effective.kv_cache_groups
    clear_cached_group_properties(kv_cache_config)
    kv_cache_config.kv_cache_tensors = tensors
    kv_cache_config.layerwise_reuse = layout
