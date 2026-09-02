from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec

_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8
KV_CACHE_TENSOR_ALIGNMENT = 2 * 1024 * 1024


def get_layerwise_physical_layer_index(layer_name: str, total_base_layers: int) -> int:
    match = re.search(
        r"(?:^|\.)mtp(?:\.layers)?\.(\d+)(?:\.|$)",
        layer_name,
    )
    if match:
        return total_base_layers + int(match.group(1))
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", layer_name)
    return int(match.group(1)) if match else 0


def get_layerwise_base_layers(physical_layers: set[int], total_base_layers: int) -> set[int]:
    """Return base-model layers, excluding MTP/spec-decode layers."""
    return {layer for layer in physical_layers if 0 <= layer < total_base_layers}


@dataclass(frozen=True)
class LayerwiseCacheLayout:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    prefetch_layer_map: dict[int, int]
    storage_indices: list[list[int]]
    has_layer_reuse: bool


@dataclass(frozen=True)
class NamedKVCacheSpec:
    layer_name: str
    spec: KVCacheSpec


@dataclass(frozen=True)
class RawCacheComponent:
    """One named cache component that can reuse an aligned raw allocation."""

    layer_name: str
    reuse_key: Hashable
    size_bytes: int
    alignment: int
    # Each view is (dtype, offset_bytes, size_bytes, shape). The current model
    # runner binds byte views here and creates backend-specific typed views in
    # its existing reshape path.
    views: tuple[tuple[torch.dtype, int, int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if self.size_bytes <= 0 or self.alignment <= 0:
            raise ValueError("Raw cache component size and alignment must be positive.")
        hash(self.reuse_key)
        for dtype, offset, size, shape in self.views:
            dtype_size = get_dtype_size(dtype)
            if (
                offset < 0
                or size <= 0
                or offset + size > self.size_bytes
                or offset % dtype_size
                or size % dtype_size
                or torch.Size(shape).numel() * dtype_size != size
            ):
                raise ValueError(f"Invalid raw cache view for {self.layer_name}.")

    def bind(self, raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if raw.numel() < self.size_bytes:
            raise ValueError(
                f"Raw cache component {self.layer_name} needs {self.size_bytes} bytes, "
                f"but its lane has {raw.numel()} bytes."
            )
        return tuple(raw[offset : offset + size].view(dtype).view(shape) for dtype, offset, size, shape in self.views)


@dataclass(frozen=True)
class LayerwiseReuseLayout:
    layer_cache_specs: dict[int, tuple[NamedKVCacheSpec, ...]]
    buffer_slots: tuple[tuple[int, ...], ...]
    component_lanes: dict[tuple[int, Hashable], tuple[RawCacheComponent, ...]]
    prefetch_layer_map: dict[int, int]
    independent_layers: list[int]
    num_prefetch_layers: int
    has_layer_reuse: bool


def get_raw_cache_components(
    layer_name: str,
    spec: KVCacheSpec,
    num_blocks: int,
) -> tuple[RawCacheComponent, ...]:
    """Normalize one named cache spec into its raw allocation component."""
    if num_blocks < 1:
        raise ValueError("num_blocks must be at least 1")

    layer_match = re.search(r"(?:^|\.)mtp(?:\.layers)?\.\d+(?:\.|$)", layer_name)
    if layer_match is None:
        layer_match = re.search(r"(?:^|\.)layers\.\d+(?:\.|$)", layer_name)
    role = layer_name[layer_match.end() :] if layer_match is not None else layer_name
    role = role.lstrip(".")
    size_bytes = spec.page_size_bytes * num_blocks

    if isinstance(spec, AscendSFAIndexerCacheSpec):
        reuse_key: Hashable = (
            role,
            "sfa_indexer",
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            spec.sfa_dcp_replicated_indexer_size,
        )
        k_size = (
            num_blocks
            * spec.sfa_dcp_replicated_indexer_size
            * spec.block_size
            * spec.num_kv_heads
            * spec.head_size
            * get_dtype_size(spec.dtype)
        )
        views: list[tuple[torch.dtype, int, int, tuple[int, ...]]] = [
            (torch.int8, 0, k_size, (k_size,)),
        ]
        if spec.scale_dim:
            scale_size = (
                num_blocks
                * spec.sfa_dcp_replicated_indexer_size
                * spec.block_size
                * spec.num_kv_heads
                * spec.scale_dim
                * get_dtype_size(spec.scale_dtype)
            )
            scale_dtype_size = get_dtype_size(spec.scale_dtype)
            scale_offset = (k_size + scale_dtype_size - 1) // scale_dtype_size * scale_dtype_size
            views.append((torch.int8, scale_offset, scale_size, (scale_size,)))
    else:
        # Non-indexer specs are deliberately conservative in the first
        # version: only equal specs with the same semantic role share storage.
        hash(spec)
        reuse_key = (role, type(spec), spec)
        views = [(torch.int8, 0, size_bytes, (size_bytes,))]

    return (
        RawCacheComponent(
            layer_name=layer_name,
            reuse_key=reuse_key,
            size_bytes=size_bytes,
            alignment=KV_CACHE_TENSOR_ALIGNMENT,
            views=tuple(views),
        ),
    )


def get_gva_layerwise_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return extra config for the MemCache GVA layerwise path."""
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
        if str(extra_config.get("backend", "mooncake")).lower() == "memcache" and extra_config.get(
            "use_layerwise", False
        ):
            return extra_config
    return None


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def build_layerwise_cache_layout(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseCacheLayout:
    shared_buffers_value = extra_config.get(_NUM_SHARED_BUFFERS) if extra_config else None
    if shared_buffers_value is None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        num_shared_buffers = num_layers
    else:
        num_shared_buffers = _parse_int_config(shared_buffers_value, _NUM_SHARED_BUFFERS)
        if num_shared_buffers < 1:
            raise ValueError(f"{_NUM_SHARED_BUFFERS} must be at least 1")

    prefetch_value = extra_config.get(_PREFETCH_LAYERS) if extra_config else None
    if prefetch_value is None:
        num_prefetch_layers = min(num_shared_buffers, _DEFAULT_MAX_PREFETCH_LAYERS)
    else:
        num_prefetch_layers = _parse_int_config(prefetch_value, _PREFETCH_LAYERS)
        if num_prefetch_layers < 1:
            raise ValueError(f"{_PREFETCH_LAYERS} must be at least 1")

    independent_value = extra_config.get(_INDEPENDENT_LAYERS) if extra_config else None
    if independent_value is None:
        layer_indices = [0]
    elif isinstance(independent_value, str) and independent_value.strip().lower() == "all":
        layer_indices = list(range(num_layers))
    elif isinstance(independent_value, list):
        layer_indices = [_parse_int_config(index, _INDEPENDENT_LAYERS) for index in independent_value]
    else:
        raise TypeError(f"{_INDEPENDENT_LAYERS} must be a list of integers or 'all'")

    normalized_indices = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            layer_index += num_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(
                f"{_INDEPENDENT_LAYERS} contains out-of-range layer index "
                f"{layer_index}; valid range is [0, {num_layers - 1}]"
            )
        normalized_indices.add(layer_index)
    independent_layers = sorted(normalized_indices)

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


def get_layerwise_kv_cache_specs(
    kv_cache_config: KVCacheConfig,
) -> dict[str, KVCacheSpec]:
    """Expand group specs into a cache spec for every logical layer."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                layer_specs[layer_name] = group_spec.kv_cache_specs[layer_name]
            else:
                layer_specs[layer_name] = group_spec
    return layer_specs


def build_layerwise_reuse_layout(
    layer_specs: dict[str, KVCacheSpec],
    total_base_layers: int,
    extra_config: dict[str, Any],
    num_blocks: int = 1,
) -> LayerwiseReuseLayout:
    """Build component lanes inside the configured physical-layer slots.

    ``num_blocks`` controls the component sizes stored in the returned layout.
    Memory planning callers use the default single-block layout, while the
    descriptor rewrite passes the globally usable block count.
    """
    # Running example legend:
    #   L=local layer, C=component, K=reuse key, A/B=buffer slot.
    #   L1..L4 correspond to local layer indices 0..3.
    #   The example uses one PP rank, so physical and local indices are equal.
    # Example input layer_specs keys:
    #   L1.C0, L1.C1, L2.C0, L2.C1,
    #   L3.C0, L3.C1, L4.C0, L4.C1, L4.C3
    named_specs_by_layer: dict[int, list[NamedKVCacheSpec]] = {}
    for layer_name, layer_spec in layer_specs.items():
        physical_layer = get_layerwise_physical_layer_index(layer_name, total_base_layers)
        named_specs_by_layer.setdefault(physical_layer, []).append(NamedKVCacheSpec(layer_name, layer_spec))
    # Example output named_specs_by_layer:
    #   L1=(C0, C1), L2=(C0, C1),
    #   L3=(C0, C1), L4=(C0, C1, C3)

    physical_layers = sorted(named_specs_by_layer)
    # Layerwise execution is local to one PP rank. Keep the global physical
    # indices only for sorting/grouping layer names, then expose contiguous
    # local execution indices to the scheduler and pool worker.
    # Example: global physical layers [2, 3, 4] map to PP-local layers
    # [0, 1, 2], where global layer 4 may be MTP layer 0.
    layer_cache_specs = {
        local_layer: tuple(
            sorted(
                named_specs_by_layer[physical_layer],
                key=lambda named_spec: named_spec.layer_name,
            )
        )
        for local_layer, physical_layer in enumerate(physical_layers)
    }
    if not layer_cache_specs:
        return LayerwiseReuseLayout(
            layer_cache_specs={},
            buffer_slots=(),
            component_lanes={},
            prefetch_layer_map={},
            independent_layers=[],
            num_prefetch_layers=0,
            has_layer_reuse=False,
        )

    # Example input: 4 layers, two shared buffers, no independent layers.
    # Example output: storage_indices=[[0, 2], [1, 3]], i.e.
    #   slot A=(L1, L3), slot B=(L2, L4).
    base_layout = build_layerwise_cache_layout(len(layer_cache_specs), extra_config)
    buffer_slots = tuple(tuple(slot) for slot in base_layout.storage_indices)
    if not base_layout.has_layer_reuse:
        return LayerwiseReuseLayout(
            layer_cache_specs=layer_cache_specs,
            buffer_slots=buffer_slots,
            component_lanes={},
            prefetch_layer_map={},
            independent_layers=list(range(len(physical_layers))),
            num_prefetch_layers=base_layout.num_prefetch_layers,
            has_layer_reuse=False,
        )

    # Example input: the slots and layer components above.
    # K0/K1/K3 are the reuse keys of components C0/C1/C3.
    # A/K0 denotes the K0 lane in buffer slot A.
    # Example output after grouping by (slot_id, reuse_key):
    #   A/K0=(L1.C0, L3.C0), A/K1=(L1.C1, L3.C1)
    #   B/K0=(L2.C0, L4.C0), B/K1=(L2.C1, L4.C1)
    #   B/K3=(L4.C3,)
    lane_components: dict[tuple[int, Hashable], list[RawCacheComponent]] = {}
    for slot_id, layers in enumerate(buffer_slots):
        for layer in layers:
            seen_keys: set[Hashable] = set()
            for named_spec in layer_cache_specs[layer]:
                # TODO: Support specs with multiple independently allocated
                # components (for example, MLA nope/rope) end to end so each
                # component can participate in a separate reuse lane.
                (component,) = get_raw_cache_components(
                    named_spec.layer_name,
                    named_spec.spec,
                    num_blocks=num_blocks,
                )
                if component.reuse_key in seen_keys:
                    raise ValueError(
                        f"Physical layer {layer} contains duplicate component reuse key {component.reuse_key!r}."
                    )
                seen_keys.add(component.reuse_key)
                lane_components.setdefault(
                    (slot_id, component.reuse_key),
                    [],
                ).append(component)
    # Freeze the collected lane members. In the example, four lanes have two
    # members, while B/K3 contains only L4.C3.
    component_lanes = {lane_key: tuple(components) for lane_key, components in lane_components.items()}
    # A slot is only a reuse candidate. Reuse is actually applied only when at
    # least one component lane has multiple members. The example returns True.
    has_layer_reuse = any(len(components) > 1 for components in component_lanes.values())

    # Validate only lanes that really share storage. Singleton lanes such as
    # B/K3 do not need cross-layer reuse support. Currently, shared lanes must
    # contain AttentionSpec components.
    # TODO: Keep unsupported shared lanes as independently allocated singleton
    # lanes, while preserving reuse for supported AttentionSpec lanes.
    for components in component_lanes.values():
        if len(components) < 2:
            continue
        unsupported = next(
            (
                component
                for component in components
                if not isinstance(
                    layer_specs[component.layer_name],
                    AttentionSpec,
                )
            ),
            None,
        )
        if unsupported is not None:
            spec = layer_specs[unsupported.layer_name]
            raise NotImplementedError(
                "Layerwise KV cache reuse supports attention cache specs only; "
                f"{unsupported.layer_name} uses {type(spec).__name__}."
            )

    # Convert component-level reuse back to slot-level runtime dependencies.
    # The example has shared lanes in both A and B, so shared_slot_ids={0, 1}.
    shared_slot_ids = {lane_key[0] for lane_key, components in component_lanes.items() if len(components) > 1}
    prefetch_layer_map: dict[int, int] = {}
    independent_layers = set(base_layout.independent_layers)
    for slot_id, slot in enumerate(buffer_slots):
        # If none of a slot's component lanes are shared, all its layers remain
        # independent and require no wait-for-previous-owner dependency.
        if slot_id not in shared_slot_ids:
            independent_layers.update(slot)
            continue
        # Chain the owners of a shared slot. For A=(L1, L3) and B=(L2, L4),
        # the example output is {L3: L1, L4: L2}, or {2: 0, 3: 1} locally.
        for owner_index in range(1, len(slot)):
            prefetch_layer_map[slot[owner_index]] = slot[owner_index - 1]

    return LayerwiseReuseLayout(
        layer_cache_specs=layer_cache_specs,
        buffer_slots=buffer_slots,
        component_lanes=component_lanes,
        prefetch_layer_map=prefetch_layer_map,
        independent_layers=sorted(independent_layers),
        num_prefetch_layers=base_layout.num_prefetch_layers,
        has_layer_reuse=has_layer_reuse,
    )


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> bool:
    """Replace per-component descriptors with one descriptor per component lane.

    Return True and update ``kv_cache_config.kv_cache_tensors`` in place when
    reuse is applied. Return False without changing the descriptors otherwise.
    """
    extra_config = get_gva_layerwise_config(vllm_config.kv_transfer_config)
    if extra_config is None:
        return False

    # Using the running example from build_layerwise_reuse_layout(), the input
    # descriptor shared_by values have one owner each:
    #   (L1.C0,), (L1.C1,), (L2.C0,), (L2.C1,),
    #   (L3.C0,), (L3.C1,), (L4.C0,), (L4.C1,), (L4.C3,)
    old_tensors = kv_cache_config.kv_cache_tensors
    if len(old_tensors) <= 1:
        return False

    # Retrieve and validate the base layers assigned to the current PP rank
    # while allowing additional MTP/spec-decode layers.
    # Example: total_base_layers=8, PP=2. On PP rank 1,
    # local_base_layers=4, (base_layer_start, base_layer_end)=(4, 8),
    # expected_base_layers={4, 5, 6, 7}, and MTP0 makes physical_layers={4, 5, 6, 7, 8}.
    local_base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    total_base_layers = vllm_config.model_config.get_total_num_hidden_layers()
    layer_specs = get_layerwise_kv_cache_specs(kv_cache_config)
    physical_layers = {get_layerwise_physical_layer_index(layer_name, total_base_layers) for layer_name in layer_specs}
    base_layer_start, base_layer_end = vllm_config.model_config.get_layers_start_end_indices(
        vllm_config.parallel_config
    )
    expected_base_layers = set(range(base_layer_start, base_layer_end))
    actual_base_layers = get_layerwise_base_layers(physical_layers, total_base_layers)
    # MTP/spec-decode layers participate in the plan, but cannot hide a missing
    # base layer or introduce a base layer owned by another PP rank.
    if actual_base_layers != expected_base_layers:
        missing_base_layers = sorted(expected_base_layers - actual_base_layers)
        unexpected_base_layers = sorted(actual_base_layers - expected_base_layers)
        logger.warning(
            "Layer reuse has missing base layers %s and unexpected base layers %s; skip tensor merge.",
            missing_base_layers,
            unexpected_base_layers,
        )
        return False
    # Build the layerwise KV cache reuse plan and count the physical layers participating in it.
    reuse_layout = build_layerwise_reuse_layout(
        layer_specs,
        total_base_layers,
        extra_config,
        num_blocks=kv_cache_config.num_blocks,
    )
    actual_layers = len(reuse_layout.layer_cache_specs)
    if not reuse_layout.has_layer_reuse:
        return False
    # The rewrite starts from one unpacked descriptor per named component. It
    # cannot safely merge descriptors that already share or slice storage.
    if any(len(tensor.shared_by) != 1 or tensor.offset != 0 or tensor.block_stride != 0 for tensor in old_tensors):
        raise NotImplementedError(
            "Layerwise KV cache reuse does not support pre-shared or packed KV cache tensor descriptors."
        )

    if actual_layers > local_base_layers:
        logger.info(
            "Layer reuse includes %d base and %d MTP/spec-decode layer(s).",
            local_base_layers,
            actual_layers - local_base_layers,
        )

    # Index the input descriptors by their sole owner, then verify that the
    # planned owner names match the input descriptor names.
    tensors_by_name = {tensor.shared_by[0]: tensor for tensor in old_tensors}
    planned_names = {
        component.layer_name for components in reuse_layout.component_lanes.values() for component in components
    }
    if planned_names != set(tensors_by_name):
        raise ValueError("Layerwise component plan does not match the KV cache tensor descriptors.")

    new_tensors: list[KVCacheTensor] = []
    # Each lane becomes one output descriptor. For example, lane A/K0 produces
    # shared_by=(L1.C0, L3.C0), while singleton B/K3 produces (L4.C3,).
    for components in reuse_layout.component_lanes.values():
        shared_by = [component.layer_name for component in components]
        cache_tensors = [tensors_by_name[layer_name] for layer_name in shared_by]
        for component, cache_tensor in zip(components, cache_tensors, strict=True):
            page_size_bytes = layer_specs[component.layer_name].page_size_bytes
            if cache_tensor.size % page_size_bytes:
                raise ValueError(
                    f"Layerwise tensor size {cache_tensor.size} for {component.layer_name} "
                    f"is not divisible by its page size {page_size_bytes}."
                )
            component_num_blocks = cache_tensor.size // page_size_bytes
            if component_num_blocks < kv_cache_config.num_blocks:
                raise ValueError(
                    f"Layerwise tensor for {component.layer_name} has "
                    f"{component_num_blocks} blocks, fewer than the configured "
                    f"minimum {kv_cache_config.num_blocks}."
                )

        # Discard per-rank surplus capacity: KVCacheManager can only use the
        # globally configured block count. Mixed layouts reserve enough bytes
        # for the largest lane member at that common block count.
        new_tensors.append(
            KVCacheTensor(
                shared_by=shared_by,
                size=max(component.size_bytes for component in components),
            )
        )
    # Example output descriptor shared_by values, one per component lane:
    #   (L1.C0, L3.C0), (L1.C1, L3.C1)
    #   (L2.C0, L4.C0), (L2.C1, L4.C1), (L4.C3)
    # Each descriptor size is the maximum size of its listed components.
    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "Layerwise KV cache reuse merged %d descriptors into %d component lanes across %d layer slots.",
        len(old_tensors),
        len(new_tensors),
        len(reuse_layout.buffer_slots),
    )
    return True
