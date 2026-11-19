# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 wire format identity and non-overlapping transfer views."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs


def cache_format_digest(config: KVCacheConfig) -> str:
    """Identify logical bytes, independently of HBM capacity or shared slots.

    Call before the scheduler unwraps uniform groups. The resulting attribute
    on KVCacheConfig survives worker IPC and scheduler configuration copying.
    """
    # v2 transfers indexer data/scale bytes exactly once in every path.
    # Older layerwise workers included the overlapping full-page view,
    # changing group offsets; those stored objects must not be reused.
    groups = []
    for group in config.kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        groups.append(
            [
                (name, type(spec).__name__, asdict(spec))
                for name in group.layer_names
                for spec in (group.kv_cache_spec.kv_cache_specs[name],)
            ]
        )
    payload = json.dumps(("dsv4-components-v2", groups), default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def transfer_cache_views(caches: dict[str, object]) -> dict[str, object]:
    """Register non-overlapping indexer payloads for whole and partial pages.

    A5 indexers expose (data, scale, full-page). Transferring all three would
    duplicate bytes and give the same logical format a different wire size.
    A token subrange selects two separate byte ranges even when a full page
    is contiguous: all data rows precede all scale rows. Keep both strides.
    """
    selected = {}
    for name, cache in caches.items():
        if not isinstance(cache, (tuple, list)) or len(cache) != 3:
            selected[name] = cache
            continue
        data, scale, full = cache
        if not name.endswith(".indexer.k_cache") or not all(isinstance(t, torch.Tensor) for t in cache):
            raise ValueError(f"Unrecognized three-component DeepSeek-V4 cache {name}.")
        stride = data.stride(0) * data.element_size()
        data_bytes = data[0].numel() * data.element_size()
        scale_bytes = scale[0].numel() * scale.element_size()
        if not (
            data.shape[:-1] == scale.shape[:-1] == full.shape[:-1]
            and all(t[0].is_contiguous() for t in cache)
            and len({t.untyped_storage().data_ptr() for t in cache}) == 1
            and full.data_ptr() == data.data_ptr()
            and scale.data_ptr() == data.data_ptr() + data_bytes
            and stride == scale.stride(0) * scale.element_size() == full.stride(0) * full.element_size()
            and full[0].numel() * full.element_size() == data_bytes + scale_bytes
            and data_bytes + scale_bytes <= stride
        ):
            raise ValueError(f"DeepSeek-V4 indexer transfer views are not disjoint payload and scales: {name}")
        selected[name] = (data, scale)
    return selected


def prompt_window_masks(worker, request, token_len: int, *, saving: bool):
    """Select physical pages for a DSV4 prompt step, preserving public checkpoint ownership.

    Return the original public masks separately so newly selected private rows
    never acquire a shareable hash. A resident draft needs no private reload.
    """
    if (
        not worker.layerwise_offload
        or getattr(worker.kv_cache_config, "dsv4_cache_format", None) is None
        or worker.cache_coordinator is None
        or worker.dcp_size != 1
        or not request.num_prompt_tokens
        or request.target_token_len > request.num_prompt_tokens
    ):
        return {}
    coordinator = worker.cache_coordinator
    original = request.store_masks if saving else request.load_masks
    selected = list(original) if original is not None else [None] * worker.num_kv_cache_groups
    public_pages = {}
    for group in worker._reused_layers_by_group:
        window = coordinator.continuation_block_range(group, token_len)
        if window is None:
            continue
        ids = (
            request.block_ids_by_group_np[group]
            if request.block_ids_by_group_np is not None
            else request.block_ids_by_group[group]
        )
        allocated = np.asarray(ids) != 0
        mask = np.zeros(len(ids), dtype=np.bool_)
        if saving:
            end = min(request.save_end_token // worker.grouped_block_size[group], len(ids))
            source = selected[group]
            if source is not None:
                end = min(end, len(source))
                mask[:end] = source[:end]
            else:
                mask[:end] = True
            public_pages[group] = mask.copy()
        start, end = window
        mask[start:end] = True
        selected[group] = mask & allocated
    if saving:
        request.store_masks = tuple(selected)
    else:
        request.load_masks = tuple(selected)
    return public_pages
