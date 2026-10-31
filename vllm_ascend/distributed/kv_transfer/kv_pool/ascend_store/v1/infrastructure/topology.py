"""Describe the resolved static topology shared by Mapping, Coordinator and Worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import regex as re
import torch.distributed as dist
from vllm.distributed import get_dcp_group, get_pcp_group, get_pp_group, get_tp_group
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    KeyMetadata,
    infer_group_cache_families,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from .configuration import KVPoolConfig


@dataclass(frozen=True, slots=True)
class TPPartitionSpec:
    """Effective Backend key partitioning for the configured TP relationship."""

    tp_mismatch: bool
    key_rank_count: int
    key_slices_per_rank: int


@dataclass(frozen=True, slots=True)
class PPPartitionSpec:
    """Target key layer spans and optional global layer reservation inside each object."""

    target_layer_counts: tuple[int, ...] | None = None
    global_layer_count: int | None = None


@dataclass(frozen=True, slots=True)
class CPPartitionSpec:
    """Context topology and object slots; local tensors are already DCP-sharded.

    Reserved slots are not writer counts. The current wire format reserves DCP
    slots only for shared objects; PCP does not multiply that reservation.
    """

    pcp_size: int = 1
    dcp_size: int = 1
    object_shard_count: int = 1
    shard_alignment_bytes: int = 1


@dataclass(frozen=True, slots=True)
class ContextRank:
    """One process's global identity and actual model-group coordinates."""

    global_rank: int
    tp_rank: int
    dcp_rank: int


@dataclass(frozen=True, slots=True)
class KVPoolLayerTopology:
    """Cache entries owned by one physical model layer inside one group."""

    physical_layer_id: int
    layer_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KVPoolGroupTopology:
    """Instance-lifetime topology for one vLLM KV cache group."""

    group_id: int
    kv_cache_spec: KVCacheSpec
    layers: tuple[KVPoolLayerTopology, ...]
    key_metadata: KeyMetadata
    is_eagle_group: bool = False
    uses_align_state: bool = False

    @property
    def block_size(self) -> int:
        return self.kv_cache_spec.block_size


@dataclass(frozen=True, slots=True)
class KVPoolTopology:
    """Process-local ranks, cache groups and layer geometry."""

    config: KVPoolConfig
    tp_rank: int
    pcp_rank: int
    groups: tuple[KVPoolGroupTopology, ...]
    transfer_groups: tuple[KVPoolGroupTopology, ...] = field(init=False)
    stage_layer_start: int
    stage_layer_count: int
    object_shard_index: int
    global_rank: int
    context_global_ranks: tuple[int, ...]
    context_replica_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        groups_by_id = {group.group_id: group for group in self.groups}
        object.__setattr__(
            self, "transfer_groups", tuple(groups_by_id[group_id] for group_id in self.config.transfer_group_ids)
        )


def resolve_worker_topology(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    config: KVPoolConfig,
) -> KVPoolTopology:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    tp_group = get_tp_group()
    tp_rank = tp_group.rank_in_group
    pp_rank = get_pp_group().rank_in_group
    pcp_size = config.cp_partition.pcp_size
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    dcp_size = config.cp_partition.dcp_size
    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0
    context_ranks: tuple[ContextRank, ...] = ()
    if config.use_layerwise and config.store_enabled:
        context_ranks = (
            _gather_context_ranks(tp_group, get_pcp_group() if pcp_size > 1 else None, dcp_rank)
            if dcp_size > 1
            else tuple(ContextRank(rank, index, 0) for index, rank in enumerate(tp_group.ranks))
        )
    head_or_tp_rank = tp_rank // config.put_step
    base_layer_count = model_config.get_total_num_hidden_layers()
    kv_cache_groups = kv_cache_config.kv_cache_groups
    group_cache_families = infer_group_cache_families(kv_cache_groups, compress_ratios=None)
    groups = []
    for group_id, group in enumerate(kv_cache_groups):
        kv_cache_spec = config.group_specs[group_id]
        uses_align_state = group_id in config.align_state_group_ids
        key_tp_rank = tp_rank if uses_align_state else head_or_tp_rank
        groups.append(
            KVPoolGroupTopology(
                group_id=group_id,
                kv_cache_spec=kv_cache_spec,
                layers=resolve_group_layers(group.layer_names, base_layer_count),
                key_metadata=KeyMetadata(
                    config.model_name,
                    key_tp_rank,
                    dcp_rank,
                    pp_rank,
                    kv_cache_group_id=group_id,
                    cache_family=group_cache_families[group_id],
                ),
                is_eagle_group=group_id in config.eagle_group_ids,
                uses_align_state=uses_align_state,
            )
        )
    stage_start = stage_count = 0
    if config.pp_partition.global_layer_count is not None:
        stage_start, stage_end = model_config.get_layers_start_end_indices(parallel_config)
        physical_layer_count = len(
            {layer.physical_layer_id for group_id in config.transfer_group_ids for layer in groups[group_id].layers}
        )
        stage_count = max(stage_end - stage_start, physical_layer_count)
    return KVPoolTopology(
        config=config,
        tp_rank=tp_rank,
        pcp_rank=pcp_rank,
        groups=tuple(groups),
        stage_layer_start=stage_start,
        stage_layer_count=stage_count,
        object_shard_index=dcp_rank if config.cp_partition.object_shard_count > 1 else 0,
        global_rank=tp_group.ranks[tp_rank],
        context_global_ranks=tuple(rank.global_rank for rank in context_ranks),
        context_replica_groups=context_replica_groups(context_ranks, tp_rank, config.put_step),
    )


def _gather_context_ranks(
    tp_group: GroupCoordinator, pcp_group: GroupCoordinator | None, dcp_rank: int
) -> tuple[ContextRank, ...]:
    """Snapshot TP x PCP membership before any background Store can use collectives.

    DCP may traverse PCP before TP. Exchange the actual coordinates instead of
    inferring them from rank arithmetic. Model CPU groups are used only during
    initialization; Store execution later owns a separate communication channel.
    """
    ranks: tuple[ContextRank, ...] = (
        ContextRank(tp_group.ranks[tp_group.rank_in_group], tp_group.rank_in_group, dcp_rank),
    )
    for group in (pcp_group, tp_group):
        if group is None or len(group.ranks) == 1:
            continue
        gathered: list[tuple[ContextRank, ...] | None] = [None] * len(group.ranks)
        dist.all_gather_object(gathered, ranks, group=group.cpu_group)
        ranks = tuple(rank for peers in gathered for rank in cast(tuple[ContextRank, ...], peers))
    return tuple(sorted(ranks, key=lambda rank: rank.global_rank))


def context_replica_groups(ranks: tuple[ContextRank, ...], tp_rank: int, put_step: int) -> tuple[tuple[int, ...], ...]:
    """Global ranks sharing a head, grouped by actual DCP coordinate; no writer election."""
    head = tp_rank // put_step
    groups: dict[int, list[int]] = {}
    for rank in ranks:
        if rank.tp_rank // put_step == head:
            groups.setdefault(rank.dcp_rank, []).append(rank.global_rank)
    return tuple(tuple(sorted(groups[coordinate])) for coordinate in sorted(groups))


def resolve_group_layers(layer_names: list[str], base_layer_count: int) -> tuple[KVPoolLayerTopology, ...]:
    names_by_physical_layer: dict[int, list[str]] = {}
    for layer_name in layer_names:
        physical_layer_id = _resolve_physical_layer_id(layer_name, base_layer_count)
        names_by_physical_layer.setdefault(physical_layer_id, []).append(layer_name)
    return tuple(
        KVPoolLayerTopology(physical_layer_id, tuple(sorted(names)))
        for physical_layer_id, names in sorted(names_by_physical_layer.items())
    )


def _resolve_physical_layer_id(layer_name: str, base_layer_count: int) -> int:
    mtp_layer = re.search(r"(?:^|\.)mtp(?:\.layers)?\.(\d+)(?:\.|$)", layer_name)
    if mtp_layer is not None:
        return base_layer_count + int(mtp_layer.group(1))
    model_layer = re.search(r"layers\.(\d+)", layer_name)
    if model_layer is not None:
        return int(model_layer.group(1))
    first_number = re.search(r"\d+", layer_name)
    if first_number is not None:
        return int(first_number.group())
    raise ValueError(f"Cannot resolve a physical layer from cache entry {layer_name!r}")
