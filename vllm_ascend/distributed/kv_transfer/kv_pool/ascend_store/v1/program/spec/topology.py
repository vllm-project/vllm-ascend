"""Define the static topology and cache geometry of one KV Pool program."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    block_size: int
    layers: tuple[KVPoolLayerTopology, ...]
    key_metadata: KeyMetadata
    uses_align_state: bool = False

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer_name for layer in self.layers for layer_name in layer.layer_names)


@dataclass(frozen=True, slots=True)
class KVPoolTopology:
    """Instance-lifetime topology and cache geometry compiled into program stages."""

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
