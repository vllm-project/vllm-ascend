# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bridge HiSparse-owned pages to the AscendStore whole-block connector."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AlignedKVCacheTensor

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker


_SLAB_ALIGNMENT = 2 * 1024 * 1024


def with_aligned_hisparse_slab(kv_cache_config: KVCacheConfig) -> KVCacheConfig:
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
            size=size + _SLAB_ALIGNMENT - 1,
            logical_size=size,
            alignment=_SLAB_ALIGNMENT,
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


def hisparse_layout_id(kv_cache_config: KVCacheConfig, dcp_size: int, transfer_granularity: int) -> str:
    # Hash planned page geometry, never addresses or pool capacity. Bump
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
    identity = (dcp_size, transfer_granularity, groups, block_strides)
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:24]


def finish_draft_host_mirror_for_save(worker: HiSparseConnectorWorker) -> None:
    """Make Core's pending draft rows durable before the external Host PUT."""
    pending = getattr(worker, "_draft_mirror_pending", None)
    if pending is False:
        return
    if pending is not True:
        raise RuntimeError("HiSparse bulk save requires Core's _draft_mirror_pending flag.")

    draft_layers = getattr(worker, "_draft_layers", None)
    enqueue_row_dma = getattr(worker, "_enqueue_row_dma", None)
    if not isinstance(draft_layers, tuple) or not draft_layers or not callable(enqueue_row_dma):
        raise RuntimeError("HiSparse bulk save requires Core's pending draft layers and row DMA hook.")
    if (
        not isinstance(getattr(worker, "_dma_submitted", None), bool)
        or getattr(worker, "host_write_event", None) is None
    ):
        raise RuntimeError("HiSparse bulk save requires Core's DMA submission flag and Host write event.")

    # Called by the bulk Store during Core's post-draft finalize_step. Core
    # ordinarily defers the draft Host copy until the next start_step, too
    # late for this step's PUT. Consume only the draft-mirror flag so Core's
    # _finish_previous_step still submits its post-forward page transfers.
    # Replace this adapter when Core exposes a public draft-row flush hook.
    compute_stream = torch.accelerator.current_stream()
    draft_ready = torch.Event()
    draft_ready.record(compute_stream)
    enqueue_row_dma(draft_layers, ready_event=draft_ready)
    worker._draft_mirror_pending = False
    if worker._dma_submitted:
        compute_stream.wait_event(worker.host_write_event)
        worker._dma_submitted = False
