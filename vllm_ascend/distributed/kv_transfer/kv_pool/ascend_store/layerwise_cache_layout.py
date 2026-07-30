from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8


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


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> None:
    """Rewrite logical layer tensors into shared physical KV cache slots."""
    extra_config = get_gva_layerwise_config(vllm_config.kv_transfer_config)
    if extra_config is None:
        return

    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    layout = build_layerwise_cache_layout(base_layers, extra_config)
    if not layout.has_layer_reuse:
        return

    if len(kv_cache_config.kv_cache_groups) != 1:
        raise NotImplementedError("Layerwise KV cache reuse requires one KV cache group.")

    old_tensors = kv_cache_config.kv_cache_tensors
    if len(old_tensors) <= 1:
        return
    if any(len(tensor.shared_by) != 1 for tensor in old_tensors):
        raise NotImplementedError("Layerwise KV cache reuse requires one KV cache tensor descriptor per layer.")
    if len(old_tensors) != base_layers:
        raise NotImplementedError("Layerwise KV cache reuse currently supports base transformer layers only.")

    layer_names = [tensor.shared_by[0] for tensor in old_tensors]
    new_tensors = []
    for slot in layout.storage_indices:
        slot_sizes = {old_tensors[index].size for index in slot}
        if len(slot_sizes) != 1:
            raise ValueError("Layers sharing a layerwise KV buffer must have equal tensor sizes.")
        new_tensors.append(
            KVCacheTensor(
                shared_by=[layer_names[index] for index in slot],
                size=old_tensors[slot[0]].size,
            )
        )
    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "Layerwise KV cache reuse merged %d tensor descriptors into %d shared slots.",
        len(old_tensors),
        len(new_tensors),
    )
