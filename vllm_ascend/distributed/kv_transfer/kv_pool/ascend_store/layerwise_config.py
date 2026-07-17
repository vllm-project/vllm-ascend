import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_EXTRA_CONFIG_KEY_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS = "layerwise_independent_layers"

# Prefetching farther than the number of shared buffers cannot improve the
# reuse pipeline. Cap the automatic value to avoid a large layer-0 burst.
_DEFAULT_MAX_PREFETCH_LAYERS = 8


def get_layerwise_physical_layer_index(layer_name: str, base_layers: int) -> int:
    """Extract the physical transformer layer index from a cache name."""
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    match = re.search(r"mtp\.(\d+)", layer_name)
    if match:
        return base_layers + int(match.group(1))
    match = re.search(r"(\d+)", layer_name)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class LayerwiseConfig:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    prefetch_layer_map: dict[int, int]
    has_layer_reuse: bool


def get_gva_layerwise_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the config for the supported GVA layerwise KV pool path."""
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
        backend = str(extra_config.get("backend", "mooncake")).lower()
        if backend == "memcache" and extra_config.get("use_layerwise", False):
            return extra_config
    return None


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def get_layerwise_num_shared_buffers(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> int:
    value = extra_config.get(_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS) if extra_config else None
    if value is None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1 to default num_shared_buffers")
        return num_layers
    num_shared_buffers = _parse_int_config(value, _EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS)
    if num_shared_buffers < 1:
        raise ValueError(f"{_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS} must be at least 1")
    return num_shared_buffers


def get_layerwise_num_prefetch_layers(
    num_shared_buffers: int,
    extra_config: dict[str, Any] | None = None,
) -> int:
    value = extra_config.get(_EXTRA_CONFIG_KEY_PREFETCH_LAYERS) if extra_config else None
    if value is None:
        return min(num_shared_buffers, _DEFAULT_MAX_PREFETCH_LAYERS)
    num_prefetch_layers = _parse_int_config(value, _EXTRA_CONFIG_KEY_PREFETCH_LAYERS)
    if num_prefetch_layers < 1:
        raise ValueError(f"{_EXTRA_CONFIG_KEY_PREFETCH_LAYERS} must be at least 1")
    return num_prefetch_layers


def _parse_layer_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(index.strip()) for index in value.split(",")]
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, Iterable):
        return [_parse_int_config(index, _EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS) for index in value]
    raise TypeError(
        f"{_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS} must be a comma-separated string or an iterable of integers"
    )


def get_layerwise_independent_layers(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> list[int]:
    value = extra_config.get(_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS) if extra_config else None
    if value is None:
        layer_indices = [0, num_layers - 1]
    elif isinstance(value, str) and value.strip().lower() == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = _parse_layer_indices(value)

    normalized_indices = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            layer_index += num_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(
                f"{_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS} contains out-of-range "
                f"layer index {layer_index}; valid range is [0, {num_layers - 1}]"
            )
        normalized_indices.add(layer_index)
    return sorted(normalized_indices)


def get_layerwise_config(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseConfig:
    num_shared_buffers = get_layerwise_num_shared_buffers(num_layers, extra_config)
    num_prefetch_layers = get_layerwise_num_prefetch_layers(num_shared_buffers, extra_config)
    independent_layers = get_layerwise_independent_layers(num_layers, extra_config)
    independent_layer_indices = set(independent_layers)
    reused_layers = [index for index in range(num_layers) if index not in independent_layer_indices]
    has_layer_reuse = len(reused_layers) > num_shared_buffers

    prefetch_layer_map = {}
    if has_layer_reuse:
        for next_index in range(num_shared_buffers, len(reused_layers)):
            prefetch_layer_map[reused_layers[next_index]] = reused_layers[next_index - num_shared_buffers]

    return LayerwiseConfig(
        num_shared_buffers=num_shared_buffers,
        num_prefetch_layers=num_prefetch_layers,
        independent_layers=independent_layers,
        prefetch_layer_map=prefetch_layer_map,
        has_layer_reuse=has_layer_reuse,
    )


def get_layerwise_kv_cache_reuse_layers(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> int | None:
    config = get_layerwise_config(num_layers, extra_config)
    return config.num_shared_buffers if config.has_layer_reuse else None


def get_layerwise_kv_cache_num_tensors(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> int | None:
    config = get_layerwise_config(num_layers, extra_config)
    if not config.has_layer_reuse:
        return None
    return len(config.independent_layers) + config.num_shared_buffers


def get_layerwise_storage_indices(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Group layer indices into independent and round-robin shared slots."""
    config = get_layerwise_config(num_layers, extra_config)
    independent_set = set(config.independent_layers)
    reused_layers = [layer for layer in range(num_layers) if layer not in independent_set]
    storage_indices = [[layer] for layer in config.independent_layers]
    for slot in range(config.num_shared_buffers):
        members = list(range(slot, len(reused_layers), config.num_shared_buffers))
        if members:
            storage_indices.append([reused_layers[index] for index in members])
    return storage_indices


def get_layer_load_start_block(
    layer_id: int,
    independent_layers: list[int],
    vllm_cached_tokens: int,
    block_size: int,
    has_layer_reuse: bool,
) -> int:
    """Return the first pool block that must be loaded for a layer."""
    if not has_layer_reuse or layer_id in independent_layers:
        return vllm_cached_tokens // block_size
    return 0
