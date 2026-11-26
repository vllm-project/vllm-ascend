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
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AlignedKVCacheTensor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import (
    DEVICE_BUFFER_ALIGNMENT,
    BufferDescriptor,
    MemoryLocation,
)

if TYPE_CHECKING:
    from vllm.config import LoadConfig, ModelConfig, VllmConfig

_HISPARSE_POOL_KEY_NAMESPACE = "hisparse-ascendstore-v1"
_HF_CONFIG_METADATA = frozenset({"_name_or_path", "transformers_version"})


def has_hisparse_cache_layout(config: KVCacheConfig | None) -> bool:
    """Detect HiSparse cache planning before or after Host pool sizing."""
    if config is None:
        return False
    host_blocks = getattr(config, "hisparse_host_num_blocks", None)
    return (isinstance(host_blocks, int) and not isinstance(host_blocks, bool)) or any(
        getattr(group, "role", None) == "hisparse_source" for group in getattr(config, "kv_cache_groups", ())
    )


def is_hisparse_ascendstore_enabled(vllm_config: Any, config: KVCacheConfig | None) -> bool:
    transfer = vllm_config.kv_transfer_config
    return has_hisparse_cache_layout(config) and transfer is not None and transfer.has_connector("AscendStoreConnector")


def _checkpoint_identity(model: ModelConfig) -> tuple[str, str | None, str | None, str, str | None]:
    # Core keeps the original URI in model_weights when downloading to a temporary directory.
    source = (model.model_weights or model.model).rstrip("/")
    if not model.model_weights and not os.path.isabs(source) and os.path.exists(source):
        source = os.path.abspath(source)
    # Hub revisions are strings such as "main" carrying the pinned commit in .resolved.
    revision = getattr(model.revision, "resolved", model.revision)
    return source, revision, model.code_revision, str(model.dtype), model.quantization


def _without_hf_config_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_hf_config_metadata(item) for key, item in value.items() if key not in _HF_CONFIG_METADATA}
    if isinstance(value, (list, tuple)):
        return [_without_hf_config_metadata(item) for item in value]
    return value


def _model_cache_identity(model: ModelConfig, load_config: LoadConfig) -> tuple[Any, ...]:
    # Read the loaded HF configs after overrides have been applied. Include the
    # text config separately because some models resolve it outside the root config.
    effective_configs = tuple(
        _without_hf_config_metadata(config.to_dict()) if config is not None else None
        for config in (getattr(model, "hf_config", None), getattr(model, "hf_text_config", None))
    )
    identity = (_checkpoint_identity(model), effective_configs, load_config.load_format == "dummy")
    if load_config.load_format in ("sharded_state", "runai_streamer_sharded"):
        # Both formats use Core's ShardedStateLoader to select files by pattern.
        extra_config = load_config.model_loader_extra_config or {}
        return (*identity, ("sharded_state", extra_config.get("pattern")))
    return identity


def build_pool_key_namespace(
    vllm_config: VllmConfig, config: KVCacheConfig | None, dcp_size: int, granularity: int
) -> str:
    """Build the external key namespace for the model and its cache layout."""
    model = vllm_config.model_config
    if not has_hisparse_cache_layout(config):
        return model.model.rstrip("/").split("/")[-1]
    assert config is not None
    speculative = vllm_config.speculative_config
    draft = speculative.draft_model_config if speculative is not None else None
    identity = (
        _model_cache_identity(model, vllm_config.load_config),
        _model_cache_identity(draft, get_draft_load_config(vllm_config)) if draft is not None else None,
        _hisparse_page_geometry(config, dcp_size, granularity),
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return f"{_HISPARSE_POOL_KEY_NAMESPACE}:{digest}"


def aligned_hisparse_allocation_config(kv_cache_config: KVCacheConfig) -> KVCacheConfig:
    """Reserve alignment padding once for the device slab shared by all device descriptors.

    Core uses their common size for one allocation. Only the device descriptors need
    replacing, so the allocation config can share the original groups and Host descriptors.
    """
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
class HiSparseTransferLayout:
    host_num_blocks: int
    host_layers: frozenset[str]
    host_storage_base_ptrs: frozenset[int]
    host_memory_group_ids: frozenset[int]

    def get_host_num_blocks(self, layer_name: str) -> int | None:
        return self.host_num_blocks if layer_name in self.host_layers else None

    def buffer_descriptors(self, regions: dict[int, tuple[int, int]]) -> list[BufferDescriptor]:
        return [
            BufferDescriptor(
                address=start,
                size_bytes=end - start,
                memory_location=MemoryLocation.HOST
                if storage in self.host_storage_base_ptrs
                else MemoryLocation.DEVICE,
                alignment_bytes=1 if storage in self.host_storage_base_ptrs else DEVICE_BUFFER_ALIGNMENT,
            )
            for storage, (start, end) in regions.items()
        ]


def plan_hisparse_store_views(
    config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Any],
    as_cache_tuple: Callable[..., tuple[torch.Tensor, ...]],
    get_storage_key: Callable[[torch.Tensor], int],
) -> tuple[HiSparseTransferLayout, dict[str, torch.Tensor], set[str]]:
    from vllm_ascend.hisparse.binding import hisparse_indexer_transfer_views

    host_num_blocks = config.hisparse_host_num_blocks
    if host_num_blocks is None:
        raise ValueError("HiSparse Host pool must be sized before AscendStore registers KV cache views.")
    excluded_layer_names = set()
    normalized_caches = {}
    host_layers = set()
    host_storage_base_ptrs: set[int] = set()
    host_memory_group_ids = set()
    for group_id, group in enumerate(config.kv_cache_groups):
        if not group.enable_kv_transfer:
            excluded_layer_names.update(group.layer_names)
            continue
        if group.host_resident:
            host_memory_group_ids.add(group_id)
        for name in group.layer_names:
            parts = as_cache_tuple(kv_caches[name])
            if group.role == "hisparse_indexer":
                parts = hisparse_indexer_transfer_views(parts, forward_context[name])
            normalized_caches[name] = parts
            if group.host_resident:
                host_layers.add(name)
                host_storage_base_ptrs.update(get_storage_key(part) for part in parts)
    layout = HiSparseTransferLayout(
        host_num_blocks, frozenset(host_layers), frozenset(host_storage_base_ptrs), frozenset(host_memory_group_ids)
    )
    return layout, normalized_caches, excluded_layer_names


def _hisparse_page_geometry(kv_cache_config: KVCacheConfig, dcp_size: int, transfer_granularity: int) -> tuple:
    """Fingerprint transferable groups and block strides for the external key namespace.

    Each layer contributes its name, dtype, tokens per block, bytes per page, head size,
    KV head count, scale dimension and scale dtype, in that order. Addresses and pool
    capacity may vary between workers. Bump the namespace for incompatible format changes.
    """
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
