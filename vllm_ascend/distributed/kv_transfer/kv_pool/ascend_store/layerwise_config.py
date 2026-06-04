from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import vllm_ascend.envs as envs_ascend

_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_EXTRA_CONFIG_KEY_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS = "layerwise_independent_layers"


@dataclass(frozen=True)
class LayerwiseConfig:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    prefetch_layer_map: dict[int, int | None]
    has_layer_reuse: bool


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
    value = None
    if extra_config and _EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS in extra_config:
        value = extra_config[_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS]
    if value is None:
        value = envs_ascend.VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS
    if value is None:
        return num_layers
    num_shared_buffers = _parse_int_config(
        value,
        _EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS,
    )
    if num_shared_buffers < 1:
        raise ValueError(
            f"{_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS} must be at least 1")
    return num_shared_buffers


def get_layerwise_num_prefetch_layers(
    extra_config: dict[str, Any] | None = None,
) -> int:
    value = None
    if extra_config and _EXTRA_CONFIG_KEY_PREFETCH_LAYERS in extra_config:
        value = extra_config[_EXTRA_CONFIG_KEY_PREFETCH_LAYERS]
    if value is None:
        value = envs_ascend.VLLM_ASCEND_KV_POOL_LAYERWISE_PREFETCH_LAYERS
    num_prefetch_layers = _parse_int_config(
        value,
        _EXTRA_CONFIG_KEY_PREFETCH_LAYERS,
    )
    if num_prefetch_layers < 1:
        raise ValueError(
            f"{_EXTRA_CONFIG_KEY_PREFETCH_LAYERS} must be at least 1")
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
        return [
            _parse_int_config(
                index, _EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS)
            for index in value
        ]
    raise TypeError(
        f"{_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS} must be a comma-separated "
        "string or an iterable of integers")


def get_layerwise_independent_layers(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> list[int]:
    value = None
    if extra_config and _EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS in extra_config:
        value = extra_config[_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS]
    if value is None:
        value = envs_ascend.VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS
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
                f"{_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS} contains "
                f"out-of-range layer index {layer_index}; valid range is "
                f"[0, {num_layers - 1}]")
        normalized_indices.add(layer_index)

    return sorted(normalized_indices)


def get_layerwise_config(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseConfig:
    num_shared_buffers = get_layerwise_num_shared_buffers(
        num_layers, extra_config)
    num_prefetch_layers = get_layerwise_num_prefetch_layers(extra_config)
    independent_layers = get_layerwise_independent_layers(
        num_layers, extra_config)
    independent_layer_indices = set(independent_layers)
    reused_layers = [
        i for i in range(num_layers)
        if i not in independent_layer_indices
    ]
    has_layer_reuse = len(reused_layers) > num_shared_buffers

    prefetch_layer_map: dict[int, int | None] = {}
    if has_layer_reuse:
        for next_index in range(num_shared_buffers, len(reused_layers)):
            prefetch_layer_map[reused_layers[next_index]] = reused_layers[
                next_index - num_shared_buffers]

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
    layerwise_config = get_layerwise_config(num_layers, extra_config)
    if not layerwise_config.has_layer_reuse:
        return None
    return layerwise_config.num_shared_buffers
