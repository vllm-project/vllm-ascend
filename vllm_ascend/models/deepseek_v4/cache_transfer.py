# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 wire format identity and non-overlapping transfer views."""

import hashlib
import json
from dataclasses import asdict

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs


def cache_format_digest(config: KVCacheConfig) -> str:
    """Identify logical bytes, independently of HBM capacity or shared slots.

    Call before the scheduler unwraps uniform groups. The resulting attribute
    on KVCacheConfig survives worker IPC and scheduler configuration copying.
    """
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
    payload = json.dumps(("dsv4-components-v1", groups), default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def transfer_cache_views(caches: dict[str, object]) -> dict[str, object]:
    """Keep data and scales once; model kernels retain the overlapping view.

    A5 indexers expose (data, scale, full-page). Transferring all three would
    duplicate bytes and give the same logical format a different wire size.
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
