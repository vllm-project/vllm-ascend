from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec

_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8


def get_layerwise_physical_layer_index(layer_name: str, base_layers: int) -> int:
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    match = re.search(r"mtp\.(\d+)", layer_name)
    if match:
        return base_layers + int(match.group(1))
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


def _get_layer_kv_cache_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
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
) -> tuple[dict[int, dict[str, str]], list[list[int]]]:
    """Build the main/indexer components assigned to each physical slot."""
    layer_layout: dict[int, dict[str, str]] = {}
    for layer_name, layer_spec in layer_specs.items():
        physical_layer = get_layerwise_physical_layer_index(layer_name, base_layers)
        component = "indexer" if isinstance(layer_spec, AscendSFAIndexerCacheSpec) else "main"
        components = layer_layout.setdefault(physical_layer, {})
        if component in components:
            raise NotImplementedError(
                f"Layerwise reuse found multiple {component} descriptors for physical layer {physical_layer}."
            )
        components[component] = layer_name

    physical_layers = sorted(layer_layout)
    if any("main" not in layer_layout[layer] for layer in physical_layers):
        raise RuntimeError("Every physical layer must provide a main KV cache for layerwise reuse.")
    layout = build_layerwise_cache_layout(len(physical_layers), extra_config)
    storage_slots = [[physical_layers[index] for index in slot] for slot in layout.storage_indices]
    return layer_layout, storage_slots


def _validate_layerwise_reuse_layout(
    kv_cache_config: KVCacheConfig,
    layer_specs: dict[str, KVCacheSpec],
    layer_layout: dict[int, dict[str, str]],
) -> None:
    """Validate topologies supported by physical layer buffer reuse."""
    if len(kv_cache_config.kv_cache_groups) == 1:
        return

    indexer_layer_names = {
        layer_name
        for layer_name, layer_spec in layer_specs.items()
        if isinstance(layer_spec, AscendSFAIndexerCacheSpec)
    }
    if not indexer_layer_names:
        raise NotImplementedError(
            "Layerwise KV cache reuse supports multiple KV cache groups only for separated SFA main/indexer caches."
        )

    unsupported_specs = {
        layer_name: type(layer_spec).__name__
        for layer_name, layer_spec in layer_specs.items()
        if not isinstance(
            layer_spec,
            (AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec),
        )
    }
    if unsupported_specs:
        details = ", ".join(f"{layer_name}={spec_name}" for layer_name, spec_name in sorted(unsupported_specs.items()))
        raise NotImplementedError(
            "Layerwise multi-group buffer reuse only supports SFA "
            f"main/indexer cache specs; unsupported cache specs: {details}."
        )

    for physical_layer, components in layer_layout.items():
        indexer_layer_name = components.get("indexer")
        if indexer_layer_name is None:
            continue
        main_layer_name = components.get("main")
        if main_layer_name is None or not isinstance(
            layer_specs[main_layer_name],
            AscendMLAAttentionSpec,
        ):
            raise RuntimeError(
                f"SFA indexer cache requires an Ascend MLA main cache at physical layer {physical_layer}."
            )


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> None:
    """Rewrite logical layer tensors into shared physical KV cache slots."""
    extra_config = get_gva_layerwise_config(vllm_config.kv_transfer_config)
    if extra_config is None:
        return

    old_tensors = kv_cache_config.kv_cache_tensors
    if len(old_tensors) <= 1:
        return
    if any(len(tensor.shared_by) != 1 for tensor in old_tensors):
        raise NotImplementedError("Layerwise KV cache reuse requires one KV cache tensor descriptor per layer.")

    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    layer_specs = _get_layer_kv_cache_specs(kv_cache_config)
    layer_layout, storage_slots = build_layerwise_reuse_layout(
        layer_specs,
        base_layers,
        extra_config,
    )
    actual_layers = len(layer_layout)
    if len(storage_slots) >= actual_layers:
        return

    _validate_layerwise_reuse_layout(
        kv_cache_config,
        layer_specs,
        layer_layout,
    )
    if actual_layers < base_layers:
        logger.warning(
            "Layer reuse expected at least %d layers, got %d; skip tensor merge.",
            base_layers,
            actual_layers,
        )
        return
    if actual_layers > base_layers:
        logger.info(
            "Layer reuse includes %d base and %d MTP/spec-decode layer(s).",
            base_layers,
            actual_layers - base_layers,
        )

    tensors_by_name = {tensor.shared_by[0]: tensor for tensor in old_tensors}
    new_tensors: list[KVCacheTensor] = []
    for physical_layers in storage_slots:
        for component in ("main", "indexer"):
            shared_by = [
                layer_layout[layer][component] for layer in physical_layers if component in layer_layout[layer]
            ]
            if not shared_by:
                continue
            component_tensors = [tensors_by_name[layer_name] for layer_name in shared_by]
            slot_sizes = {tensor.size for tensor in component_tensors}
            if len(slot_sizes) != 1:
                raise ValueError(f"Layers sharing a layerwise slot must have equal {component} tensor sizes.")
            reference_spec = layer_specs[shared_by[0]]
            if any(layer_specs[layer_name] != reference_spec for layer_name in shared_by[1:]):
                raise ValueError(f"Layers sharing a layerwise slot must have identical {component} cache specs.")
            new_tensors.append(
                KVCacheTensor(
                    shared_by=shared_by,
                    size=component_tensors[0].size,
                )
            )
    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "Layerwise KV cache reuse merged %d descriptors into %d descriptors across %d physical buffers.",
        len(old_tensors),
        len(new_tensors),
        len(storage_slots),
    )
