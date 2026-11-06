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
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.core.layerwise_cache_config import (
    LayerwisePolicy,
    LayerwiseReuseLayout,
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
        reuse_names = getattr(group, "reuse_layer_names", None)
        for name in group.layer_names:
            spec = (
                group.kv_cache_spec.kv_cache_specs[name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
            if reuse_names is not None:
                spec = replace(spec, reuse=name in reuse_names)
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


def _descriptor_buckets(domains, specs, base_layers):
    buckets = []
    for tensors in domains:
        signatures = []
        for tensor in tensors:
            for name in tensor.layers:
                spec = specs[name]
                if not spec.reuse:
                    continue
                layer = get_physical_layer_index(name, base_layers)
                for signature, stride, members in signatures:
                    if (
                        type(spec) is type(signature)
                        and spec == signature
                        and tensor.block_stride == stride
                        and all(get_physical_layer_index(member, base_layers) != layer for member in members)
                    ):
                        members.append(name)
                        break
                else:
                    signatures.append((spec, tensor.block_stride, [name]))
        buckets.extend(members for _, _, members in signatures)
    return buckets


def build_layerwise_reuse_layout(
    kv_cache_config: KVCacheConfig,
    base_layers: int,
    extra_config: dict[str, Any],
    *,
    excluded_layer_names: set[str] | None = None,
) -> LayerwiseReuseLayout:
    """Infer slots inside original backing domains; never infer storage from groups."""
    specs = get_layerwise_kv_cache_specs(kv_cache_config)
    layers = sorted({get_physical_layer_index(name, base_layers) for name in specs})
    policy = normalize_layerwise_policy(layers, extra_config, tuple(sorted(excluded_layer_names or ())))
    for name, spec in specs.items():
        if spec.reuse and (
            name in policy.excluded_owners or get_physical_layer_index(name, base_layers) in policy.independent_layers
        ):
            raise ValueError(f"reuse=True conflicts with excluded/independent configuration: {name}")
    for group in kv_cache_config.kv_cache_groups:
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
    previous = get_layerwise_layout(kv_cache_config)
    if previous is not None:
        if (
            replace(previous.policy, excluded_owners=policy.excluded_owners) != policy
            or previous.num_blocks != kv_cache_config.num_blocks
        ):
            raise ValueError("Layerwise policy/capacity changed after placement")
        requested_names = {name for name, spec in specs.items() if spec.reuse}
        if requested_names != {name for slot in previous.buffer_slots for name in slot}:
            raise ValueError("Layerwise reuse intent changed after placement")
        if previous.has_layer_reuse:
            try:
                valid = build_layerwise_tensors(kv_cache_config, previous) == kv_cache_config.kv_cache_tensors
            except (ValueError, NotImplementedError):
                valid = False
            if not valid:
                raise ValueError("Layerwise descriptors changed after placement")
        return previous

    if not any(spec.reuse for spec in specs.values()):
        return LayerwiseReuseLayout(policy, kv_cache_config.num_blocks, (), frozenset(specs), {})
    domains = _storage_domains(kv_cache_config, specs)
    buckets = _descriptor_buckets(domains, specs, base_layers)
    shared_names = {
        name for tensors in domains if tensors[0].backing_id is not None for t in tensors for name in t.layers
    }

    group_ids = {name: i for i, group in enumerate(kv_cache_config.kv_cache_groups) for name in group.layer_names}
    slots = []
    predecessors = {}
    for bucket in buckets:
        ordered = sorted(bucket, key=lambda name: (get_physical_layer_index(name, base_layers), name))
        for index in range(min(policy.num_shared_buffers, len(ordered))):
            slot = tuple(ordered[index :: policy.num_shared_buffers])
            slots.append(slot)
            last_by_lifetime = {}
            for name in slot:
                # Explicit shared-pool descriptors retain the original group
                # block-ID lifetimes. Unrelated allocations never imply aliases.
                lifetime = group_ids[name] if name in shared_names else 0
                layer = get_physical_layer_index(name, base_layers)
                if lifetime in last_by_lifetime:
                    source = last_by_lifetime[lifetime]
                    if source >= layer:
                        raise ValueError(f"Simultaneous cache regions cannot alias: {name}")
                    predecessors.setdefault(layer, set()).add(source)
                last_by_lifetime[lifetime] = layer
    rotating = {name for slot in slots if len(slot) > 1 for name in slot}
    for group in kv_cache_config.kv_cache_groups:
        transfer_units = {}
        for name in group.layer_names:
            if name in policy.excluded_owners:
                continue
            layer = get_physical_layer_index(name, base_layers)
            transfer_units.setdefault(layer, set()).add(name in rotating)
        if any(len(flags) > 1 for flags in transfer_units.values()):
            raise NotImplementedError(
                "Mixed rotating/resident components within one group/layer transfer are unsupported"
            )
    return LayerwiseReuseLayout(
        policy,
        kv_cache_config.num_blocks,
        tuple(slots),
        frozenset(name for name, spec in specs.items() if not spec.reuse),
        {layer: tuple(sorted(sources)) for layer, sources in predecessors.items()},
    )


def _pack_shared_domain(config, tensors, layout, specs):
    """Pack resident banks and rotating slots, retaining proven pool aliases."""
    by_name = {name: tensor for tensor in tensors for name in tensor.layers}
    regions = _layer_regions(tensors, config.num_blocks)
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
    placed = []
    for slot in slots:
        stride = by_name[slot[0]].block_stride
        groups = {group_ids[name] for name in slot}
        source_regions = {regions[name] for name in slot}
        for bank_stride, bank_groups, bank_regions, offset in placed:
            if bank_stride == stride and groups.isdisjoint(bank_groups) and source_regions & bank_regions:
                bank_groups.update(groups)
                bank_regions.update(source_regions)
                break
        else:
            offset = cursor
            cursor += config.num_blocks * stride
            placed.append((stride, set(groups), set(source_regions), offset))
        offsets.update(dict.fromkeys(slot, offset))

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
        # Bucketing already requires one backing, equal concrete specs and
        # equal physical block strides. All members have the same slot offset.
        # Group membership remains in kv_cache_groups, indexed by layer name;
        # it does not require splitting this physical view into descriptors.
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
    if not kv_cache_config.kv_cache_tensors:
        if any(spec.reuse for spec in get_layerwise_kv_cache_specs(kv_cache_config).values()):
            raise ValueError("Requested layerwise reuse has no original cache descriptors")
        return
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    layout = build_layerwise_reuse_layout(
        kv_cache_config,
        base_layers,
        extra,
        excluded_layer_names=excluded_layer_names,
    )
    if get_layerwise_layout(kv_cache_config) is not None:
        return
    tensors = (
        build_layerwise_tensors(kv_cache_config, layout) if layout.has_layer_reuse else kv_cache_config.kv_cache_tensors
    )
    names = [name for tensor in tensors for name in get_kv_cache_tensor_layers(tensor)]
    if len(names) != len(set(names)) or set(names) != set(get_layerwise_kv_cache_specs(kv_cache_config)):
        raise ValueError("Layerwise conversion lost or duplicated cache names")
    if any(tensor.layer_stride < 0 or tensor.block_stride <= 0 or tensor.offset < 0 for tensor in tensors):
        raise ValueError("Layerwise conversion produced an invalid descriptor")
    kv_cache_config.kv_cache_tensors = tensors
    kv_cache_config.layerwise_reuse = layout
