"""Static formulas for the Mooncake Layerwise KeyRange data plane."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import block_hash_to_str

from ...topology import KVPoolTopology
from ..reachability import HybridReachability, UnitaryReachability, bind_layerwise_reachability

ByteArray: TypeAlias = NDArray[np.uint64]
KeyAxes: TypeAlias = tuple[tuple[str, ...], ...]
LayerRangeBatch: TypeAlias = tuple[ByteArray, ByteArray, ByteArray]
LayerwiseFullKey: TypeAlias = Callable[[int, str, int, int], str]
Reachability: TypeAlias = UnitaryReachability | HybridReachability


@dataclass(frozen=True, slots=True)
class KeyRangeLayerLayout:
    """Pre-sliced arrays used by one physical layer's terminal formula."""

    base_addresses: ByteArray
    block_lengths: ByteArray
    block_strides: ByteArray
    object_offsets: ByteArray


@dataclass(frozen=True, slots=True)
class KeyRangeLayerwiseGroupProjection:
    """Registration-bound KeyRange facts for one cache group."""

    group_id: int
    block_size: int
    hash_block_size: int
    physical_layer_ids: tuple[int, ...]
    local_key_coordinates: tuple[tuple[int, int], ...]
    lookup_key_coordinates: tuple[tuple[int, int], ...]
    make_full_key: LayerwiseFullKey
    object_size: int
    layers: Mapping[int, KeyRangeLayerLayout]


@dataclass(frozen=True, slots=True)
class KeyRangeLayerwiseProjection:
    """All immutable formulas needed by one KeyRange Layerwise route."""

    group_ids: tuple[int, ...]
    groups: Mapping[int, KeyRangeLayerwiseGroupProjection]
    reachability: Reachability


@dataclass(frozen=True, slots=True)
class KeyRangeLayerwiseProjectionBinder:
    """Bind KeyRange layout arrays when the Worker registers its KV cache."""

    topology: KVPoolTopology
    max_model_len: int
    full_key: LayerwiseFullKey
    use_eagle: bool = False
    retention_interval: int | None = None
    reachability: Reachability = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reachability",
            bind_layerwise_reachability(
                self.topology,
                self.max_model_len,
                use_eagle=self.use_eagle,
                retention_interval=self.retention_interval,
            ),
        )

    def bind(
        self,
        base_addresses: Mapping[int, Sequence[int]],
        block_lengths: Mapping[int, Sequence[int]],
        block_strides: Mapping[int, Sequence[int]],
        layer_entry_offsets: Mapping[int, Sequence[int]],
    ) -> KeyRangeLayerwiseProjection:
        groups = {
            group.group_id: _bind_key_range_group(
                self.topology,
                group.group_id,
                self.full_key,
                base_addresses,
                block_lengths,
                block_strides,
                layer_entry_offsets,
            )
            for group in self.topology.transfer_groups
        }
        return KeyRangeLayerwiseProjection(
            self.topology.transfer_group_ids,
            MappingProxyType(groups),
            self.reachability,
        )


def key_range_local_keys(
    group: KeyRangeLayerwiseGroupProjection,
    hashes: Sequence[BlockHash | str],
) -> KeyAxes:
    """Evaluate the fixed local key axes for dynamic content hashes."""

    hash_strings = tuple(block_hash_to_str(value) for value in hashes)
    return tuple(
        tuple(group.make_full_key(group.group_id, value, head_rank, pp_rank) for value in hash_strings)
        for head_rank, pp_rank in group.local_key_coordinates
    )


def key_range_lookup_keys(
    group: KeyRangeLayerwiseGroupProjection,
    hashes: Sequence[BlockHash | str],
) -> KeyAxes:
    """Evaluate all producer coordinates required by one Lookup."""

    hash_strings = tuple(block_hash_to_str(value) for value in hashes)
    return tuple(
        tuple(group.make_full_key(group.group_id, value, head_rank, pp_rank) for value in hash_strings)
        for head_rank, pp_rank in group.lookup_key_coordinates
    )


def key_range_layer_ranges(
    group: KeyRangeLayerwiseGroupProjection,
    block_ids: Sequence[int] | ByteArray,
    token_counts: Sequence[int] | ByteArray,
    *,
    layer_id: int,
) -> LayerRangeBatch:
    """Map dynamic object rows directly to final KeyRange layer arrays."""

    ids, counts = _dynamic_rows(block_ids, token_counts)
    layout = group.layers[layer_id]
    addresses = layout.base_addresses[None, :] + ids[:, None] * layout.block_strides[None, :]
    sizes = layout.block_lengths[None, :] * counts[:, None] // np.uint64(group.block_size)
    offsets = np.broadcast_to(layout.object_offsets, addresses.shape)
    return addresses, sizes, offsets


def _bind_key_range_group(
    topology: KVPoolTopology,
    group_id: int,
    full_key: LayerwiseFullKey,
    base_addresses: Mapping[int, Sequence[int]],
    block_lengths: Mapping[int, Sequence[int]],
    block_strides: Mapping[int, Sequence[int]],
    layer_entry_offsets: Mapping[int, Sequence[int]],
) -> KeyRangeLayerwiseGroupProjection:
    group = next(group for group in topology.transfer_groups if group.group_id == group_id)
    try:
        bases = _readonly(base_addresses[group_id])
        lengths = _readonly(block_lengths[group_id])
        strides = _readonly(block_strides[group_id])
        entry_offsets = layer_entry_offsets[group_id]
    except KeyError as error:
        raise RuntimeError(f"KV cache group {group_id} has not registered local memory") from error
    physical_layer_ids = tuple(layer.physical_layer_id for layer in group.layers)
    _validate_registration(group_id, bases, lengths, strides, entry_offsets, physical_layer_ids)

    offsets: ByteArray = np.empty(len(lengths), dtype=np.uint64)
    offsets[0] = np.uint64(0)
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    offsets.flags.writeable = False
    layers = MappingProxyType(
        {
            layer_id: KeyRangeLayerLayout(
                bases[start:end],
                lengths[start:end],
                strides[start:end],
                offsets[start:end],
            )
            for layer_index, layer_id in enumerate(physical_layer_ids)
            for start, end in ((int(entry_offsets[layer_index]), int(entry_offsets[layer_index + 1])),)
        }
    )
    metadata = group.key_metadata
    lookup_coordinates = tuple(
        (head_rank, pp_rank)
        for pp_rank in range(topology.pp_size)
        for head_rank in range(topology.tp_partition.key_rank_count)
    )
    return KeyRangeLayerwiseGroupProjection(
        group_id,
        group.block_size,
        topology.hash_block_size,
        physical_layer_ids,
        ((metadata.head_or_tp_rank, metadata.pp_rank),),
        lookup_coordinates,
        full_key,
        int(lengths.sum(dtype=np.uint64)),
        layers,
    )


def _validate_registration(
    group_id: int,
    bases: ByteArray,
    lengths: ByteArray,
    strides: ByteArray,
    layer_offsets: Sequence[int],
    physical_layer_ids: tuple[int, ...],
) -> None:
    if len(bases) == 0 or not (len(bases) == len(lengths) == len(strides)):
        raise ValueError(f"KV cache group {group_id} registered misaligned memory arrays")
    if len(layer_offsets) != len(physical_layer_ids) + 1 or layer_offsets[0] != 0 or layer_offsets[-1] != len(bases):
        raise ValueError(f"KV cache group {group_id} registered invalid layer entry bounds")


def _dynamic_rows(
    block_ids: Sequence[int] | ByteArray,
    token_counts: Sequence[int] | ByteArray,
) -> tuple[ByteArray, ByteArray]:
    ids = np.asarray(block_ids, dtype=np.uint64)
    counts = np.asarray(token_counts, dtype=np.uint64)
    if len(counts) != len(ids):
        raise ValueError("Block IDs and token counts must describe the same rows")
    return ids, counts


def _readonly(values: Sequence[int]) -> ByteArray:
    result = np.array(values, dtype=np.uint64, copy=True)
    result.flags.writeable = False
    return result
