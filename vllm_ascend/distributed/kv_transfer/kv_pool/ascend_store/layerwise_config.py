from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import vllm_ascend.envs as envs_ascend


@dataclass(frozen=True)
class LayerwiseConfig:
    num_shared_buffers: int
    independent_layers: list[int]
    shared_layers: list[int]
    buffer_owner_layers: list[int]
    prefetch_layer_map: dict[int, tuple[int, int]]


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def get_layerwise_num_shared_buffers() -> int:
    value = envs_ascend.VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS
    num_shared_buffers = _parse_int_config(
        value,
        "VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS",
    )
    if num_shared_buffers < 1:
        raise ValueError("VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS must be at least 1")
    return num_shared_buffers


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
            _parse_int_config(index, "VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS")
            for index in value
        ]
    raise TypeError(
        "VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS must be a comma-separated string "
        "or an iterable of integers")


def get_layerwise_independent_layers(num_layers: int) -> list[int]:
    value = envs_ascend.VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS
    if value is None:
        layer_indices = [0, num_layers - 1]
    else:
        layer_indices = _parse_layer_indices(value)

    normalized_indices = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            layer_index += num_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(
                "VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS contains out-of-range "
                f"layer index {layer_index}; valid range is [0, {num_layers - 1}]"
            )
        normalized_indices.add(layer_index)

    return sorted(normalized_indices)


def get_layerwise_config(num_layers: int) -> LayerwiseConfig:
    num_shared_buffers = get_layerwise_num_shared_buffers()
    independent_layers = get_layerwise_independent_layers(num_layers)
    if len(independent_layers) == num_layers:
        raise ValueError(
            "VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS cannot include "
            "all layers; at least one layer must use layerwise KV pool transfer")
    independent_layer_indices = set(independent_layers)
    shared_layers = [i for i in range(num_layers)
                     if i not in independent_layer_indices]
    buffer_owner_layers = shared_layers[:num_shared_buffers]

    prefetch_layer_map = {}
    for current_index in range(1, len(shared_layers)):
        previous_index = current_index - 1
        next_index = previous_index + num_shared_buffers
        if next_index < len(shared_layers):
            prefetch_layer_map[shared_layers[current_index]] = (
                shared_layers[previous_index],
                shared_layers[next_index],
            )

    return LayerwiseConfig(
        num_shared_buffers=num_shared_buffers,
        independent_layers=independent_layers,
        shared_layers=shared_layers,
        buffer_owner_layers=buffer_owner_layers,
        prefetch_layer_map=prefetch_layer_map,
    )
