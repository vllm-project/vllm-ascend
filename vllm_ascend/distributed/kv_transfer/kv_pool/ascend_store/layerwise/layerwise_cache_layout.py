from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_backend_class,
)
from vllm_ascend.models.cache_reuse import get_cache_reuse_planner

_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8


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
class LayerwiseCacheConfig:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    has_layer_reuse: bool

    @classmethod
    def parse(cls, num_layers: int, extra_config: dict[str, Any] | None = None) -> LayerwiseCacheConfig:
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

        return cls(
            num_shared_buffers=num_shared_buffers,
            num_prefetch_layers=num_prefetch_layers,
            independent_layers=independent_layers,
            has_layer_reuse=num_layers - len(independent_layers) > num_shared_buffers,
        )


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


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


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


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    *,
    excluded_layer_names: set[str] | None = None,
) -> None:
    """Apply the model's plan; exclusions affect reuse, not cache identity."""
    layout = getattr(kv_cache_config, "layerwise_reuse_layout", None)
    if layout is not None:
        if excluded_layer_names and layout.reused_cache_names.intersection(excluded_layer_names):
            raise ValueError("Reuse exclusions must be supplied before the cache layout is planned.")
        return
    extra_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if extra_config is None or not kv_cache_config.kv_cache_tensors:
        return
    planner = get_cache_reuse_planner(vllm_config)
    if planner is None:
        base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        if LayerwiseCacheConfig.parse(base_layers, extra_config).has_layer_reuse:
            raise NotImplementedError("Layerwise offload requires a registered model reuse planner.")
        return
    planner(kv_cache_config, vllm_config, extra_config, excluded_layer_names).apply()
