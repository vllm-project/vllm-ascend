from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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
    value = extra_config.get(_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS) if extra_config else None
    if value is None:
        return num_layers
    num_shared_buffers = _parse_int_config(value, _EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS)
    if num_shared_buffers < 1:
        raise ValueError(f"{_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS} must be at least 1")
    return num_shared_buffers


def get_layerwise_num_prefetch_layers(
    extra_config: dict[str, Any] | None = None,
) -> int:
    value = extra_config.get(_EXTRA_CONFIG_KEY_PREFETCH_LAYERS) if extra_config else None
    if value is None:
        return 1
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
                f"{_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS} contains "
                f"out-of-range layer index {layer_index}; valid range is "
                f"[0, {num_layers - 1}]"
            )
        normalized_indices.add(layer_index)

    return sorted(normalized_indices)


def get_layerwise_config(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseConfig:
    num_shared_buffers = get_layerwise_num_shared_buffers(num_layers, extra_config)
    num_prefetch_layers = get_layerwise_num_prefetch_layers(extra_config)
    independent_layers = get_layerwise_independent_layers(num_layers, extra_config)
    independent_layer_indices = set(independent_layers)
    reused_layers = [i for i in range(num_layers) if i not in independent_layer_indices]
    has_layer_reuse = len(reused_layers) > num_shared_buffers

    prefetch_layer_map: dict[int, int | None] = {}
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
    layerwise_config = get_layerwise_config(num_layers, extra_config)
    if not layerwise_config.has_layer_reuse:
        return None
    return layerwise_config.num_shared_buffers


def get_layerwise_kv_cache_num_tensors(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> int | None:
    """Number of distinct KV cache buffers after layer reuse.

    Returns ``None`` when layer reuse is disabled (one buffer per layer).
    Otherwise returns the count of merged tensors the model runner will
    allocate: each independent layer keeps its own buffer, and the reused
    layers share ``num_shared_buffers`` buffers. The worker uses this to size
    the memory-inflation factor so total allocation stays within budget.
    """
    config = get_layerwise_config(num_layers, extra_config)
    if not config.has_layer_reuse:
        return None
    return len(config.independent_layers) + config.num_shared_buffers


def get_layerwise_storage_indices(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Group layer indices into shared storage slots for layer reuse.

    Each inner list holds the layer indices that time-multiplex one shared
    buffer. Independent layers each occupy their own slot; the reused layers
    are distributed across ``num_shared_buffers`` slots round-robin so that
    ``reused_layers[k]`` shares a buffer with
    ``reused_layers[k + num_shared_buffers]`` — matching the prefetch map
    computed in :func:`get_layerwise_config`.
    """
    config = get_layerwise_config(num_layers, extra_config)
    independent_set = set(config.independent_layers)
    reused_layers = [layer for layer in range(num_layers) if layer not in independent_set]
    storage_indices: list[list[int]] = [[layer] for layer in config.independent_layers]
    for slot in range(config.num_shared_buffers):
        members = list(range(slot, len(reused_layers), config.num_shared_buffers))
        if members:
            storage_indices.append([reused_layers[m] for m in members])
    return storage_indices


def get_layer_load_start_block(
    layer_id: int,
    independent_layers: list[int],
    vllm_cached_tokens: int,
    block_size: int,
    has_layer_reuse: bool,
) -> int:
    """First pool block to load for ``layer_id``.

    Without layer reuse every layer behaves the same and starts right after
    the HBM-cached blocks. With layer reuse, independent layers still skip
    HBM-cached blocks (their dedicated buffer keeps that KV valid), while
    shared (time-multiplexed) layers must reload every block from block 0
    because HBM hits are not reliable.
    """
    if not has_layer_reuse or layer_id in independent_layers:
        return vllm_cached_tokens // block_size
    return 0
