# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bridge HiSparse-owned pages to the AscendStore whole-block connector."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AlignedKVCacheTensor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import DEVICE_BUFFER_ALIGNMENT, BufferDescriptor

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

_HISPARSE_CACHE_NAMESPACE = "hisparse-v4"


def has_hisparse_layout(config: KVCacheConfig | None) -> bool:
    """Use the same layout predicate during connector setup and view binding."""
    if config is None:
        return False
    host_blocks = getattr(config, "hisparse_host_num_blocks", None)
    return (isinstance(host_blocks, int) and not isinstance(host_blocks, bool)) or any(
        getattr(group, "role", None) == "hisparse_source" for group in getattr(config, "kv_cache_groups", ())
    )


def uses_hisparse_store(vllm_config: Any, config: KVCacheConfig | None) -> bool:
    transfer = vllm_config.kv_transfer_config
    return has_hisparse_layout(config) and transfer is not None and transfer.has_connector("AscendStoreConnector")


def _checkpoint_identity(model: ModelConfig) -> tuple[str, str | None, str | None, str, str | None]:
    # Core keeps the original URI in model_weights when downloading to a temporary directory.
    source = (model.model_weights or model.model).rstrip("/")
    if not model.model_weights and not os.path.isabs(source) and os.path.exists(source):
        source = os.path.abspath(source)
    return source, model.revision, model.code_revision, str(model.dtype), model.quantization


def store_model_name(vllm_config: VllmConfig, config: KVCacheConfig | None, dcp_size: int, granularity: int) -> str:
    model = vllm_config.model_config
    if not has_hisparse_layout(config):
        return model.model.rstrip("/").split("/")[-1]
    assert config is not None
    speculative = vllm_config.speculative_config
    draft = speculative.draft_model_config if speculative is not None else None
    identity = (
        _checkpoint_identity(model),
        _checkpoint_identity(draft) if draft is not None else None,
        _hisparse_page_geometry(config, dcp_size, granularity),
    )
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:24]
    return f"{_HISPARSE_CACHE_NAMESPACE}:{digest}"


def aligned_hisparse_allocation_config(kv_cache_config: KVCacheConfig) -> KVCacheConfig:
    """Pass the original size through Core's config copies on the padded descriptors."""
    original_tensors = kv_cache_config.kv_cache_tensors
    device_sizes = {
        tensor.logical_size if isinstance(tensor, AlignedKVCacheTensor) else tensor.size
        for tensor in original_tensors
        if not tensor.host_resident
    }
    if len(device_sizes) != 1:
        raise ValueError(f"HiSparse alignment requires one shared device slab size, got {sorted(device_sizes)}.")
    (size,) = device_sizes
    allocation_config = copy(kv_cache_config)
    allocation_config.kv_cache_tensors = [
        t
        if t.host_resident
        else AlignedKVCacheTensor(
            **{f.name: getattr(t, f.name) for f in fields(KVCacheTensor) if f.init and f.name != "size"},
            size=size + DEVICE_BUFFER_ALIGNMENT - 1,
            logical_size=size,
            alignment=DEVICE_BUFFER_ALIGNMENT,
        )
        for t in original_tensors
    ]
    return allocation_config


@dataclass(frozen=True)
class HiSparseStoreViews:
    host_num_blocks: int
    host_layers: frozenset[str]
    host_storage_keys: frozenset[int]
    host_groups: frozenset[int]

    def block_count(self, layer_name: str) -> int | None:
        return self.host_num_blocks if layer_name in self.host_layers else None

    def buffer_descriptors(self, regions: dict[int, tuple[int, int]]) -> list[BufferDescriptor]:
        return [
            BufferDescriptor(
                address=start,
                size_bytes=end - start,
                host=storage in self.host_storage_keys,
                alignment_bytes=1 if storage in self.host_storage_keys else DEVICE_BUFFER_ALIGNMENT,
            )
            for storage, (start, end) in regions.items()
        ]


def plan_hisparse_store_views(
    config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Any],
    as_cache_tuple: Callable[..., tuple[torch.Tensor, ...]],
    get_storage_key: Callable[[torch.Tensor], int],
) -> tuple[HiSparseStoreViews, dict[str, torch.Tensor], set[str]]:
    from vllm_ascend.hisparse.binding import hisparse_indexer_transfer_views

    host_num_blocks = config.hisparse_host_num_blocks
    assert host_num_blocks is not None
    excluded = set()
    normalized = {}
    host_layers = set()
    host_storage_keys: set[int] = set()
    host_groups = set()
    for group_id, group in enumerate(config.kv_cache_groups):
        if not group.enable_kv_transfer:
            excluded.update(group.layer_names)
            continue
        if group.host_resident:
            host_groups.add(group_id)
        for name in group.layer_names:
            parts = as_cache_tuple(kv_caches[name])
            if group.role == "hisparse_indexer":
                parts = hisparse_indexer_transfer_views(parts, forward_context[name])
            normalized[name] = parts
            if group.host_resident:
                host_layers.add(name)
                host_storage_keys.update(get_storage_key(part) for part in parts)
    layout = HiSparseStoreViews(
        host_num_blocks, frozenset(host_layers), frozenset(host_storage_keys), frozenset(host_groups)
    )
    return layout, normalized, excluded


def _hisparse_page_geometry(kv_cache_config: KVCacheConfig, dcp_size: int, transfer_granularity: int) -> tuple:
    # Include planned page geometry, never addresses or pool capacity. Bump
    # the key version for any serialization change outside these fields.
    groups = []
    layers = set()
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if not group.enable_kv_transfer:
            continue
        group_layers = []
        for name in sorted(group.layer_names):
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            group_layers.append(
                (
                    name,
                    str(getattr(spec, "dtype", getattr(spec, "dtypes", None))),
                    spec.block_size,
                    spec.page_size_bytes,
                    getattr(spec, "head_size", None),
                    getattr(spec, "num_kv_heads", None),
                    getattr(spec, "scale_dim", 0),
                    str(getattr(spec, "scale_dtype", None)),
                )
            )
            layers.add(name)
        groups.append((group_id, group.role, group.host_resident, group_layers))
    block_strides = sorted(
        (name, tensor.block_stride)
        for tensor in kv_cache_config.kv_cache_tensors
        for name in tensor.layers
        if name in layers
    )
    return dcp_size, transfer_granularity, groups, block_strides
