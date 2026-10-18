"""Describe the resolved static topology shared by Mapping, Reachability and Worker."""

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
    pcp_size: int
    dcp_size: int
    put_step: int
    cache_transfer_granularity: int
    hash_block_size: int
    tp_partition: TPPartitionSpec
    groups: tuple[KVPoolGroupTopology, ...]
    transfer_group_ids: tuple[int, ...]
    consumer_pipeline_partitions: tuple[int, ...] | None

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
