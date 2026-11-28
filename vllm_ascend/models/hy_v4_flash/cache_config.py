# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""The HY4 packed KV row is shared by the full and sliding attention layers."""

from collections.abc import Sequence
from math import gcd

from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec, MLAAttentionSpec, SlidingWindowMLASpec

PACKED_CACHE_FORMAT = "hy4_packed"


def is_hy4_packed_cache(spec: KVCacheSpec) -> bool:
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) and spec.cache_dtype_str == PACKED_CACHE_FORMAT


def hy4_required_pages(spec: KVCacheSpec, prompt_tokens: int) -> range:
    """Logical pages for a handoff that may replay the final prompt token."""
    window = getattr(spec, "sliding_window", None)
    first_token = max(0, prompt_tokens - window) if window is not None else 0
    return range(first_token // spec.block_size, (prompt_tokens + spec.block_size - 1) // spec.block_size)


def hy4_transfer_layout(groups: Sequence[KVCacheGroupSpec]) -> list[dict] | None:
    if not groups or not all(is_hy4_packed_cache(group.kv_cache_spec) for group in groups):
        return None
    return [
        {
            "layers": list(group.layer_names),
            "block_size": group.kv_cache_spec.block_size,
            "page_bytes": group.kv_cache_spec.page_size_bytes,
            "dtype": str(group.kv_cache_spec.dtype),
            "window": getattr(group.kv_cache_spec, "sliding_window", None),
        }
        for group in groups
    ]


def group_hy4_packed_cache(specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    full = [name for name, spec in specs.items() if isinstance(spec, MLAAttentionSpec)]
    sliding = [name for name, spec in specs.items() if isinstance(spec, SlidingWindowMLASpec)]
    group_size = gcd(len(full), len(sliding))
    if len({spec.page_size_bytes for spec in specs.values()}) != 1:
        raise ValueError("HY4 packed cache groups require equal physical page sizes.")

    groups = []
    for names in (full, sliding):
        for start in range(0, len(names), group_size):
            layers = names[start : start + group_size]
            spec = specs[layers[0]]
            if any(specs[name] != spec for name in layers):
                raise ValueError("HY4 packed cache layers in a group must have identical specs.")
            groups.append(KVCacheGroupSpec(layers, spec))
    return groups
