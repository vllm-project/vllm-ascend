# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from .model import DeepseekV41Topology


@dataclass(frozen=True)
class CacheResource:
    cache_name: str
    owner_layer: int
    group_id: int


@dataclass(frozen=True)
class LayerCacheAccess:
    execution_layer: int
    reads: tuple[CacheResource, ...]
    writes: tuple[CacheResource, ...]
    topk_source_layer: int | None
    candidate_source_layer: int | None


def build_layer_cache_access_plan(
    topology: "DeepseekV41Topology",
    cache_config: "KVCacheConfig",
    model_prefix: str,
) -> tuple[LayerCacheAccess, ...]:
    """Resolve the full backbone's cache dependencies against the final layout.

    Writes describe potential writes. The executing forward path determines
    which requests and ranges were actually updated.
    """
    group_by_name = {
        name: group_id
        for group_id, group in enumerate(cache_config.kv_cache_groups)
        for name in group.layer_names
    }
    root = f"{model_prefix}." if model_prefix else ""

    def cache_name(owner: int, suffix: str) -> str:
        return f"{root}layers.{owner}.self_attn.{suffix}"

    def resource(owner: int, suffix: str) -> CacheResource:
        name = cache_name(owner, suffix)
        if name not in group_by_name:
            raise ValueError(f"DeepSeek V4.1 layerwise cache is missing from KVCacheConfig: {name}")
        return CacheResource(name, owner, group_by_name[name])

    plan = []
    for role in topology.layers:
        layer = role.layer_idx
        swa = resource(layer, "swa_cache")
        reads = [swa]
        writes = [swa]
        owner = role.kv_source_layer
        if owner is not None:
            long_kv = resource(owner, "long_kv_cache")
            reads.append(long_kv)
            if role.is_kv_source:
                writes.append(long_kv)
            if role.is_index_source:
                index_k = resource(owner, "indexer.k_cache")
                reads.append(index_k)
                if role.is_kv_source:
                    writes.append(index_k)
            if cache_name(owner, "indexer.k_cache_folded") in group_by_name:
                folded = resource(owner, "indexer.k_cache_folded")
                if role.is_index_source and role.uses_candidate_filter:
                    reads.append(folded)
                if role.is_kv_source:
                    writes.append(folded)
            if role.is_kv_source and role.compress_ratio == 2:
                state = resource(layer, "compressor.state_cache")
                reads.append(state)
                writes.append(state)

        plan.append(
            LayerCacheAccess(
                execution_layer=layer,
                reads=tuple(reads),
                writes=tuple(writes),
                topk_source_layer=role.index_source_layer,
                candidate_source_layer=topology.candidate_source_layer_id if role.uses_candidate_filter else None,
            )
        )
    return tuple(plan)
