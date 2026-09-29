from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_layerwise_protocol,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers

_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8
_INDEXER_CACHE_SUFFIX = ".indexer.k_cache"


def get_layerwise_physical_layer_index(layer_name: str, base_layers: int) -> int:
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


@dataclass(frozen=True)
class NamedKVCacheSpec:
    layer_name: str
    spec: KVCacheSpec


@dataclass(frozen=True)
class LayerwiseLayerCacheSpecs:
    main: NamedKVCacheSpec
    indexer: NamedKVCacheSpec | None = None
    extra_main_specs: tuple[NamedKVCacheSpec, ...] = ()


@dataclass(frozen=True)
class LayerwiseReuseLayout:
    layer_cache_specs: dict[int, LayerwiseLayerCacheSpecs]
    buffer_slots: tuple[tuple[int, ...], ...]
    prefetch_layer_map: dict[int, int]
    independent_layers: list[int]
    num_prefetch_layers: int
    has_layer_reuse: bool


def get_layerwise_reuse_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the extra config of the layerwise-reuse connector, if any.

    A connector opts into layerwise reuse when its backend carries a
    layerwise protocol and the protocol accepts the connector's extra
    config. Both checks resolve through the backend registry — the generic
    layer never names the protocol or the backend.
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
        protocol = get_layerwise_protocol(str(extra_config.get("backend", "mooncake")))
        if protocol is None:
            continue
        layerwise_config = protocol.extract_layout_config(extra_config)
        if layerwise_config is not None:
            return layerwise_config
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
    base_layers: int,
    extra_config: dict[str, Any],
) -> LayerwiseReuseLayout:
    """Build reusable physical-layer slots by grouping layers on their main cache spec."""
    named_specs_by_layer: dict[int, list[NamedKVCacheSpec]] = {}
    for layer_name, layer_spec in layer_specs.items():
        physical_layer = get_layerwise_physical_layer_index(layer_name, base_layers)
        named_specs_by_layer.setdefault(physical_layer, []).append(NamedKVCacheSpec(layer_name, layer_spec))

    physical_layers = sorted(named_specs_by_layer)
    base_layout = build_layerwise_cache_layout(len(physical_layers), extra_config)
    independent_layers = [physical_layers[index] for index in base_layout.independent_layers]
    independent_layer_set = set(independent_layers)

    layer_cache_specs: dict[int, LayerwiseLayerCacheSpecs] = {}
    for physical_layer, named_specs in named_specs_by_layer.items():
        if len(named_specs) == 1:
            layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(main=named_specs[0])
            continue

        indexer_specs = [spec for spec in named_specs if spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        main_specs = [spec for spec in named_specs if not spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        if len(main_specs) < 1:
            raise ValueError(
                f"Physical layer {physical_layer} has no main cache spec; "
                f"got {[spec.layer_name for spec in named_specs]}."
            )
        # Select '.attn' as main spec, rest as extra
        main_spec = next((s for s in main_specs if s.layer_name.endswith(".attn")), main_specs[0])
        indexer_spec = indexer_specs[0] if indexer_specs else None
        # Keep every remaining named component. ``extra_main_specs`` predates
        # component lanes, but it is also the least invasive place to retain a
        # second indexer or any other component that is neither the selected
        # main nor the primary indexer.
        extra_specs = tuple(s for s in named_specs if s is not main_spec and s is not indexer_spec)
        layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(
            main=main_spec,
            indexer=indexer_spec,
            extra_main_specs=extra_specs,
        )

    signature_buckets: list[tuple[KVCacheSpec, list[int]]] = []
    for physical_layer in physical_layers:
        if physical_layer in independent_layer_set:
            continue
        # TODO(lf): Plan shared buffers independently for every cache spec.
        # Slots are grouped by main spec. Indexer specs are validated separately.
        signature = layer_cache_specs[physical_layer].main.spec
        for bucket_signature, bucket_layers in signature_buckets:
            if signature == bucket_signature:
                bucket_layers.append(physical_layer)
                break
        else:
            signature_buckets.append((signature, [physical_layer]))

    buffer_slots: list[tuple[int, ...]] = [(layer,) for layer in independent_layers]
    prefetch_layer_map: dict[int, int] = {}
    for _, bucket_layers in signature_buckets:
        num_shared_buffers = min(base_layout.num_shared_buffers, len(bucket_layers))
        for buffer_index in range(num_shared_buffers):
            layers_sharing_buffer = tuple(bucket_layers[buffer_index::num_shared_buffers])
            buffer_slots.append(layers_sharing_buffer)
            for owner_index in range(1, len(layers_sharing_buffer)):
                prefetch_layer_map[layers_sharing_buffer[owner_index]] = layers_sharing_buffer[owner_index - 1]

    if prefetch_layer_map:
        unsupported_specs = [
            named_spec
            for named_specs in named_specs_by_layer.values()
            for named_spec in named_specs
            if not isinstance(named_spec.spec, AttentionSpec)
        ]
        if unsupported_specs:
            named_spec = unsupported_specs[0]
            raise NotImplementedError(
                "Layerwise KV cache reuse supports attention cache specs only; "
                f"{named_spec.layer_name} uses {type(named_spec.spec).__name__}."
            )

    return LayerwiseReuseLayout(
        layer_cache_specs=layer_cache_specs,
        buffer_slots=tuple(buffer_slots),
        prefetch_layer_map=prefetch_layer_map,
        independent_layers=independent_layers,
        num_prefetch_layers=base_layout.num_prefetch_layers,
        has_layer_reuse=bool(prefetch_layer_map),
    )


def iter_layerwise_buffer_lanes(
    reuse_layout: LayerwiseReuseLayout,
) -> tuple[tuple[NamedKVCacheSpec, ...], ...]:
    """Group aliasing named specs into one lane per physical KV buffer.

    Successor of the retired ``LayerwiseReuseLayout.component_lanes`` field:
    each returned lane lists the named specs that alias one physical buffer
    inside a slot. DSV4 MLA mains and SFA indexers share per-role buffers,
    identical generic attention specs share per role, and any other spec
    (e.g. compressor state caches) stays private to its own buffer.
    """
    lane_components: dict[tuple[int, tuple[Any, ...]], list[NamedKVCacheSpec]] = {}
    for slot_id, slot in enumerate(reuse_layout.buffer_slots):
        for physical_layer in slot:
            entry = reuse_layout.layer_cache_specs[physical_layer]
            named_specs = (entry.main, *entry.extra_main_specs)
            if entry.indexer is not None:
                named_specs += (entry.indexer,)
            for named_spec in named_specs:
                role_match = re.search(
                    r"(?:^|\.)(?:mtp(?:\.layers)?|layers)\.\d+\.",
                    named_spec.layer_name,
                )
                role = (
                    named_spec.layer_name[role_match.end():]
                    if role_match is not None
                    else named_spec.layer_name
                )
                spec = named_spec.spec
                if isinstance(spec, AscendMLAAttentionSpec) and spec.model_version == "deepseek_v4":
                    reuse_key = ("deepseek_v4_contiguous_raw", role)
                elif isinstance(spec, AscendSFAIndexerCacheSpec):
                    # BF16 and C8 indexers can interpret different prefixes of
                    # one contiguous raw allocation. Geometry that changes the
                    # logical indexer shape remains part of the compatibility
                    # key; dtype/scale layout deliberately does not.
                    reuse_key = (
                        "sfa_indexer_contiguous_raw",
                        role,
                        spec.block_size,
                        spec.num_kv_heads,
                        spec.head_size,
                        spec.sfa_dcp_replicated_indexer_size,
                    )
                elif isinstance(spec, AttentionSpec):
                    reuse_key = ("identical_attention_spec", role, spec)
                else:
                    # State caches are preserved but remain private until
                    # their allocation representation has an explicit reuse
                    # contract.
                    reuse_key = ("private", named_spec.layer_name)
                lane_components.setdefault((slot_id, reuse_key), []).append(named_spec)
    return tuple(tuple(components) for components in lane_components.values())


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> None:
    """Rewrite logical layer tensors to use shared physical KV buffers."""
    extra_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if extra_config is None:
        return

    old_tensors = kv_cache_config.kv_cache_tensors
    if not old_tensors:
        return

    local_base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    total_base_layers = vllm_config.model_config.get_total_num_hidden_layers()
    layer_specs = get_layerwise_kv_cache_specs(kv_cache_config)
    physical_layers = {get_layerwise_physical_layer_index(layer_name, total_base_layers) for layer_name in layer_specs}
    base_layer_start, base_layer_end = vllm_config.model_config.get_layers_start_end_indices(
        vllm_config.parallel_config
    )
    expected_base_layers = set(range(base_layer_start, base_layer_end))
    actual_base_layers = {layer for layer in physical_layers if 0 <= layer < total_base_layers}
    if actual_base_layers != expected_base_layers:
        logger.warning(
            "Layer reuse has missing base layers %s and unexpected base layers %s; skip tensor merge.",
            sorted(expected_base_layers - actual_base_layers),
            sorted(actual_base_layers - expected_base_layers),
        )
        return False

    reuse_layout = build_layerwise_reuse_layout(
        layer_specs,
        total_base_layers,
        extra_config,
    )
    actual_layers = len(reuse_layout.layer_cache_specs)
    if not reuse_layout.has_layer_reuse:
        return
    if actual_layers < local_base_layers:
        logger.warning(
            "Layer reuse expected at least %d layers, got %d; skip tensor merge.",
            local_base_layers,
            actual_layers,
        )
        return
    if actual_layers > local_base_layers:
        logger.info(
            "Layer reuse includes %d base and %d MTP/spec-decode layer(s).",
            local_base_layers,
            actual_layers - local_base_layers,
        )

    # vLLM describes multiple contiguous layer regions in one backing
    # allocation. Reuse replaces that placement before any storage exists;
    # each new descriptor owns one physical slot whose layers alias it.
    seen_layers: set[str] = set()
    for tensor in old_tensors:
        for layer_idx, layer_name in enumerate(get_kv_cache_tensor_layers(tensor)):
            spec = layer_specs[layer_name]
            layer_size = kv_cache_config.num_blocks * spec.page_size_bytes
            start = tensor.offset + layer_idx * tensor.layer_stride
            if tensor.block_stride != spec.page_size_bytes:
                raise NotImplementedError("Layerwise KV cache reuse requires contiguous per-layer pages.")
            if start < 0 or start + layer_size > tensor.size:
                raise ValueError(f"Layerwise KV cache descriptor for {layer_name} exceeds its backing allocation.")
            if layer_name in seen_layers:
                raise ValueError(f"Duplicate layerwise KV cache descriptor for {layer_name}.")
            seen_layers.add(layer_name)
    if seen_layers != set(layer_specs):
        raise ValueError("Layerwise KV cache descriptors must cover every cache spec.")

    def _merge_specs(named_specs: list[NamedKVCacheSpec]) -> None:
        shared_by = [named_spec.layer_name for named_spec in named_specs]
        reference_spec = layer_specs[shared_by[0]]
        if any(layer_specs[layer_name] != reference_spec for layer_name in shared_by[1:]):
            raise ValueError(
                "Layers sharing layerwise KV buffers must have identical cache specs for every named cache spec."
            )
        new_tensors.append(
            KVCacheTensor(
                layers=shared_by,
                size=kv_cache_config.num_blocks * reference_spec.page_size_bytes,
                layer_stride=0,
                block_stride=reference_spec.page_size_bytes,
                offset=0,
            )
        )
    planned_names = {
        named_spec.layer_name
        for lane in iter_layerwise_buffer_lanes(reuse_layout)
        for named_spec in lane
    }
    if planned_names != set(layer_specs):
        missing = sorted(set(layer_specs) - planned_names)
        unexpected = sorted(planned_names - set(layer_specs))
        raise ValueError(
            f"Layerwise component plan does not match the cache specs; missing={missing}, unexpected={unexpected}."
        )

    new_tensors: list[KVCacheTensor] = []
    for lane in iter_layerwise_buffer_lanes(reuse_layout):
        _merge_specs(list(lane))
    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "Layerwise KV cache reuse merged %d descriptors into %d descriptors using %d buffer assignments.",
        len(old_tensors),
        len(new_tensors),
        len(new_tensors),
    )
    return True
