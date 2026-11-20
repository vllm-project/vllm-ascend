# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 wire format identity and non-overlapping transfer views."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs


def bind_pool_format(cache_config: KVCacheConfig, vllm_config) -> None:
    """Bind the existing GVA wire identity for resident and rotating producers."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
        get_layerwise_gva_config,
    )
    from vllm_ascend.models.deepseek_v4.cache_reuse import uses_dspark_push

    if get_layerwise_gva_config(vllm_config.kv_transfer_config) is None:
        return
    cache_config.dsv4_cache_format = cache_format_digest(cache_config)
    if uses_dspark_push(vllm_config):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import DSparkPrefixKeys

        keys = DSparkPrefixKeys.from_config(vllm_config)
        assert keys is not None
        cache_config.dsv4_dspark_push = True
        cache_config.dsv4_cache_format += f"-dspark-{keys.identity}"


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


def select_window_pages(block_ids, window, checkpoint_mask=None, *, public_end: int | None = None):
    """Return the live transfer mask and, for saves, the shareable subset.

    Window bounds and public_end are exclusive page indices. A physical block
    ID of zero represents an unallocated page, even at a retained checkpoint.
    """
    allocated = np.asarray(block_ids) != 0
    live = np.zeros(len(allocated), dtype=np.bool_)
    public = None
    if public_end is not None:
        end = min(public_end, len(live))
        if checkpoint_mask is not None:
            end = min(end, len(checkpoint_mask))
            live[:end] = checkpoint_mask[:end]
        else:
            live[:end] = True
        public = live & allocated
    start, end = window
    live[start:end] = True
    return live & allocated, public


def select_prompt_window_masks(blocks, block_sizes, original_masks, windows, *, save_end_token=None):
    """Keep other groups' masks intact. Only groups with a continuation window change."""
    masks = list(original_masks) if original_masks is not None else [None] * len(block_sizes)
    public_pages = {}
    for group, window in windows:
        if window is None:
            continue
        end = save_end_token // block_sizes[group] if save_end_token is not None else None
        masks[group], public = select_window_pages(blocks[group], window, masks[group], public_end=end)
        if public is not None:
            public_pages[group] = public
    return tuple(masks), public_pages
