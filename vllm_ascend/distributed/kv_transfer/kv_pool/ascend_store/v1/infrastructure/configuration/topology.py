"""Describe the resolved static topology shared by Mapping, Coordinator and Worker."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata


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


def context_replica_groups(ranks: tuple[ContextRank, ...], tp_rank: int, put_step: int) -> tuple[tuple[int, ...], ...]:
    """Global ranks sharing a head, grouped by actual DCP coordinate; no writer election."""
    head = tp_rank // put_step
    groups: dict[int, list[int]] = {}
    for rank in ranks:
        if rank.tp_rank // put_step == head:
            groups.setdefault(rank.dcp_rank, []).append(rank.global_rank)
    return tuple(tuple(sorted(groups[coordinate])) for coordinate in sorted(groups))


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

    @property
    def block_size(self) -> int:
        return self.kv_cache_spec.block_size

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer_name for layer in self.layers for layer_name in layer.layer_names)

    @property
    def uses_align_state(self) -> bool:
        return kv_cache_spec_uses_align_state(self.kv_cache_spec)


@dataclass(frozen=True, slots=True)
class KVPoolTopology:
    """Instance-lifetime topology and cache geometry available to every owner."""

    tp_rank: int
    tp_size: int
    pp_size: int
    pcp_rank: int
    put_step: int
    cache_transfer_granularity: int
    hash_block_size: int
    tp_partition: TPPartitionSpec
    pp_partition: PPPartitionSpec
    cp_partition: CPPartitionSpec
    groups: tuple[KVPoolGroupTopology, ...]
    transfer_group_ids: tuple[int, ...]
    stage_layer_start: int
    stage_layer_count: int
    object_shard_index: int
    global_rank: int
    context_global_ranks: tuple[int, ...]
    context_replica_groups: tuple[tuple[int, ...], ...]

    @property
    def consumer_pipeline_partitions(self) -> tuple[int, ...] | None:
        return self.pp_partition.target_layer_counts

    @property
    def pcp_size(self) -> int:
        return self.cp_partition.pcp_size

    @property
    def dcp_size(self) -> int:
        return self.cp_partition.dcp_size

    @property
    def transfer_groups(self) -> tuple[KVPoolGroupTopology, ...]:
        groups_by_id = {group.group_id: group for group in self.groups}
        try:
            return tuple(groups_by_id[group_id] for group_id in self.transfer_group_ids)
        except KeyError as error:
            raise ValueError(f"Unknown transferable KV cache group {error.args[0]}") from error


def kv_cache_spec_uses_align_state(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether an upstream cache spec contains mutable Mamba align state."""

    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = kv_cache_spec.kv_cache_specs.values()
    else:
        specs = (kv_cache_spec,)
    return any(isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align" for spec in specs)
