# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Prefix-cache participation semantics for heterogeneous KV caches."""

from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)


def kv_cache_spec_participates_in_prefix_caching(spec: KVCacheSpec) -> bool:
    """Whether *spec* may publish and reuse prefix-cache blocks.

    Upstream KV-cache specs participate by default. Platform-specific caches
    can opt out with a ``participates_in_prefix_caching = False`` class
    attribute. A uniform group must not mix the two lifetimes because it owns
    one block table and one prefix-cache manager.
    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        participation = {
            kv_cache_spec_participates_in_prefix_caching(member) for member in spec.kv_cache_specs.values()
        }
        if len(participation) > 1:
            raise ValueError(
                "UniformTypeKVCacheSpecs cannot mix prefix-cache-participating and non-participating cache specs."
            )
        return participation.pop() if participation else True
    return bool(getattr(spec, "participates_in_prefix_caching", True))


def kv_cache_group_participates_in_prefix_caching(
    group: KVCacheGroupSpec,
) -> bool:
    """Whether a KV-cache group participates in prefix caching."""
    return kv_cache_spec_participates_in_prefix_caching(group.kv_cache_spec)


def prefix_cache_group_ids(groups: list[KVCacheGroupSpec]) -> tuple[int, ...]:
    """Return group IDs that take part in hash publication and lookup."""
    return tuple(
        group_id for group_id, group in enumerate(groups) if kv_cache_group_participates_in_prefix_caching(group)
    )
