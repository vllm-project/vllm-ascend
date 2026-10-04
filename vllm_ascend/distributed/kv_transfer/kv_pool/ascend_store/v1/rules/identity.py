"""Define the logical identity of transferable KV objects.

Identity rules determine:

- which token range forms one transferable logical object;
- which content hash and local Block ID identify its local data;
- which model, cache Group, cache family, and parallel coordinates identify it
  across ranks and the Backend;
- which rank owns and publishes it;
- how checkpoint and consumer-PP variants retain the same identity relation;
- which Backend key encodes the resulting identity.

In short, identity determines what the remote object is, who handles it, and
which local data source it represents. The key is the final encoding of that
identity. Physical byte ranges and request lifetime remain outside this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    PoolKey,
    block_hash_to_str,
    get_block_hashes,
)

from ..topology import KVPoolTopology

ByteArray: TypeAlias = NDArray[np.uint64]
ChunkRows: TypeAlias = tuple[ByteArray, ByteArray, tuple[BlockHash | str, ...]]
BlockRows: TypeAlias = tuple[ByteArray, tuple[BlockHash | str, ...], ByteArray]
KeyAxes: TypeAlias = tuple[tuple[str, ...], ...]
LayerwiseFullKey = Callable[[int, str, int, int], str]
LayerwisePartialKey = Callable[..., str]


# =============================================================================
# Lookup and Transfer Rows
# =============================================================================
#
# Bind normal or Hybrid lookup and tail behavior from fixed cache geometry, then
# map request hashes and Block Tables into caller-owned rows for Load and Store.


def bind_chunk_rules(topology: KVPoolTopology):
    normal: dict[int, Callable] = {}
    lookup: dict[int, Callable] = {}
    blocks: dict[int, Callable] = {}
    fine_grained_lookup = any(
        group.uses_align_state and group.block_size > topology.hash_block_size for group in topology.transfer_groups
    )
    for group in topology.transfer_groups:
        cacheable = group.group_id in topology.transfer_group_ids
        normal[group.group_id] = partial(
            _chunk_rows, block_size=group.block_size, hash_block_size=topology.hash_block_size, cacheable=cacheable
        )
        if fine_grained_lookup:
            lookup[group.group_id] = cast(
                Callable,
                partial(
                    _fine_lookup_rows,
                    block_size=group.block_size,
                    hash_block_size=topology.hash_block_size,
                    cacheable=cacheable,
                ),
            )
        else:
            lookup[group.group_id] = normal[group.group_id]
        blocks[group.group_id] = partial(
            _block_rows,
            block_size=group.block_size,
            hash_block_size=topology.hash_block_size,
            minimum_block_id=0 if topology.tp_partition.tp_mismatch or not group.uses_align_state else 1,
            cacheable=cacheable,
        )
    return partial(_dispatch, normal), partial(_dispatch, lookup), partial(_dispatch, blocks)


def _chunk_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    *,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
    block_size: int,
    hash_block_size: int,
    cacheable: bool,
) -> ChunkRows:
    if not cacheable or not block_hashes or end_token <= 0:
        return _empty_chunk_rows()
    grouped_hashes = get_block_hashes(block_hashes, block_size, hash_block_size)
    logical_count = min(len(grouped_hashes), _ceil_div(end_token, block_size))
    aligned_start_token = start_token // block_size * block_size
    starts: list[int] = []
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    for block_index in range(logical_count):
        start = block_index * block_size
        end = min(start + block_size, end_token)
        if start < aligned_start_token or end <= start:
            continue
        if mask is not None and (block_index >= len(mask) or not mask[block_index]):
            continue
        starts.append(start)
        counts.append(end - start)
        hashes.append(grouped_hashes[block_index])
    return _readonly(starts), _readonly(counts), tuple(hashes)


def _fine_lookup_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    *,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
    block_size: int,
    hash_block_size: int,
    cacheable: bool,
) -> ChunkRows:
    if not cacheable or not block_hashes or end_token <= 0:
        return _empty_chunk_rows()
    first_hash = start_token // hash_block_size
    last_hash = min(len(block_hashes), end_token // hash_block_size)
    starts: list[int] = []
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    for hash_index in range(first_hash, last_hash):
        boundary = (hash_index + 1) * hash_block_size
        block_index = _ceil_div(boundary, block_size) - 1
        if mask is not None and (block_index >= len(mask) or not mask[block_index]):
            continue
        block_start = block_index * block_size
        starts.append(block_start)
        counts.append(boundary - block_start)
        hashes.append(block_hashes[hash_index])
    return _readonly(starts), _readonly(counts), tuple(hashes)


def _block_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    block_ids: Sequence[int],
    *,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
    tail_boundary_token: int | None = None,
    block_size: int,
    hash_block_size: int,
    minimum_block_id: int,
    cacheable: bool,
) -> BlockRows:
    if not cacheable or not block_hashes or end_token <= 0:
        return _empty_block_rows()
    grouped_hashes = get_block_hashes(block_hashes, block_size, hash_block_size)
    requested_block_count = _ceil_div(end_token, block_size)
    tail_block_index = None
    if tail_boundary_token is not None:
        tail_block_index = requested_block_count - 1
        if tail_boundary_token < end_token or _ceil_div(tail_boundary_token, block_size) != requested_block_count:
            raise ValueError(
                f"Tail-key boundary {tail_boundary_token} does not identify the Load tail ending at {end_token}"
            )
    logical_count = (
        requested_block_count if tail_block_index is not None else min(len(grouped_hashes), requested_block_count)
    )
    block_offset = max(logical_count - len(block_ids), 0)
    aligned_start_token = start_token // block_size * block_size
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    selected_block_ids: list[int] = []
    for block_index in range(logical_count):
        start = block_index * block_size
        end = min(start + block_size, end_token)
        if start < aligned_start_token or end <= start:
            continue
        if mask is not None and (block_index >= len(mask) or not mask[block_index]):
            continue
        local_index = block_index - block_offset
        if local_index < 0 or local_index >= len(block_ids):
            continue
        block_id = block_ids[local_index]
        if block_id < minimum_block_id:
            continue
        # Read the (possibly lazily grouped) hash only after every cheap
        # request-side exclusion has succeeded.
        if tail_block_index is not None and block_index == tail_block_index:
            assert tail_boundary_token is not None
            hashes.append(_boundary_hash(tail_boundary_token, block_hashes, hash_block_size))
        else:
            if block_index >= len(grouped_hashes):
                continue
            hashes.append(grouped_hashes[block_index])
        counts.append(end - start)
        selected_block_ids.append(block_id)
    return _readonly(counts), tuple(hashes), _readonly(selected_block_ids)


# =============================================================================
# State Checkpoint Rows
# =============================================================================
#
# Preserve the exact handed-off state Block and add only the companion KV Blocks
# required to publish the same token boundary.


def bind_checkpoint_rule(topology: KVPoolTopology, ownership: Callable):
    align_state_group_ids = tuple(group.group_id for group in topology.transfer_groups if group.uses_align_state)
    if not align_state_group_ids:
        return None
    block_sizes = {group.group_id: group.block_size for group in topology.transfer_groups}
    minimum_block_ids = {
        group.group_id: 1 if group.uses_align_state and not topology.tp_partition.tp_mismatch else 0
        for group in topology.transfer_groups
    }
    companion_group_ids = tuple(
        group.group_id for group in topology.transfer_groups if group.group_id not in align_state_group_ids
    )
    return partial(
        _checkpoint_rows,
        checkpoint_group_ids=align_state_group_ids,
        companion_group_ids=companion_group_ids,
        block_sizes=block_sizes,
        minimum_block_ids=minimum_block_ids,
        hash_block_size=topology.hash_block_size,
        ownership=ownership,
    )


def _checkpoint_rows(
    boundary_token: int,
    block_hashes: Sequence[BlockHash | str],
    source_block_ids: Mapping[int, int],
    block_ids_by_group: Mapping[int, Sequence[int]],
    *,
    published_store_end_token: int = 0,
    checkpoint_group_ids: tuple[int, ...],
    companion_group_ids: tuple[int, ...],
    block_sizes: Mapping[int, int],
    minimum_block_ids: Mapping[int, int],
    hash_block_size: int,
    ownership: Callable,
) -> tuple[tuple[int, BlockRows], ...]:
    checkpoint_hash = _boundary_hash(boundary_token, block_hashes, hash_block_size)
    rows_by_group: list[tuple[int, BlockRows]] = []
    source_groups = tuple(group_id for group_id in checkpoint_group_ids if group_id in source_block_ids)
    accepted_source_groups: list[int] = []
    for group_id in source_groups:
        block_size = block_sizes[group_id]
        block_index = _ceil_div(boundary_token, block_size) - 1
        block_id = source_block_ids[group_id]
        if block_id < minimum_block_ids[group_id]:
            continue
        accepted_source_groups.append(group_id)
        rows: BlockRows = (_readonly((block_size,)), (checkpoint_hash,), _readonly((block_id,)))
        rows_by_group.append((group_id, ownership(group_id, rows)))

    if accepted_source_groups and any(boundary_token % block_sizes[group_id] for group_id in accepted_source_groups):
        for group_id in companion_group_ids:
            block_size = block_sizes[group_id]
            last_block_index = _ceil_div(boundary_token, block_size) - 1
            first_block_index = min(published_store_end_token // block_size, last_block_index)
            block_ids = block_ids_by_group[group_id]
            logical_count = last_block_index + 1
            block_offset = max(logical_count - len(block_ids), 0)
            counts: list[int] = []
            hashes: list[BlockHash | str] = []
            selected_ids: list[int] = []
            for block_index in range(first_block_index, logical_count):
                local_index = block_index - block_offset
                if local_index < 0 or local_index >= len(block_ids):
                    continue
                if block_ids[local_index] < minimum_block_ids[group_id]:
                    continue
                block_end = min((block_index + 1) * block_size, boundary_token)
                counts.append(block_size)
                hashes.append(_boundary_hash(block_end, block_hashes, hash_block_size))
                selected_ids.append(block_ids[local_index])
            rows = (_readonly(counts), tuple(hashes), _readonly(selected_ids))
            rows_by_group.append((group_id, ownership(group_id, rows)))
    return tuple(rows_by_group)


# =============================================================================
# Store Ownership
# =============================================================================
#
# Bind the responsible rank from the parallel layout, then shard only the valid
# Store candidates that remain after request-side filtering.


def bind_store_ownership(topology: KVPoolTopology):
    selectors: dict[int, Callable] = {}
    for group in topology.transfer_groups:
        replica_count = topology.put_step
        if topology.tp_partition.tp_mismatch or topology.dcp_size > 1 or group.uses_align_state:
            replica_count = 1
        shard_rank = topology.pcp_rank * replica_count + topology.tp_rank % replica_count
        shard_count = topology.pcp_size * replica_count
        selectors[group.group_id] = partial(_owned_rows, shard_rank=shard_rank, shard_count=shard_count)
    return partial(_dispatch, selectors)


def _owned_rows(rows: BlockRows, *, shard_rank: int, shard_count: int) -> BlockRows:
    if shard_count <= 1:
        return rows
    counts, hashes, block_ids = rows
    selection = slice(shard_rank, len(counts), shard_count)
    return counts[selection], hashes[selection], block_ids[selection]


# =============================================================================
# Backend Key Identity
# =============================================================================
#
# Precompute local, lookup, and consumer-pipeline key coordinates so request-time
# calls only append content hashes or request-specific partial-key fields.


def resolve_store_pipeline_ranks(topology: KVPoolTopology) -> dict[int, tuple[int, ...]] | None:
    partitions = topology.consumer_pipeline_partitions
    if partitions is None or len(partitions) <= 1:
        return None
    partition_bounds = []
    first_layer = 0
    for layer_count in partitions:
        last_layer = first_layer + layer_count
        partition_bounds.append((first_layer, last_layer))
        first_layer = last_layer
    return {
        group.group_id: tuple(
            pp_rank
            for pp_rank, (first_layer, last_layer) in enumerate(partition_bounds)
            if any(first_layer <= layer.physical_layer_id < last_layer for layer in group.layers)
        )
        for group in topology.transfer_groups
    }


def bind_key_rules(
    topology: KVPoolTopology,
    layerwise_full_key: LayerwiseFullKey | None,
    layerwise_partial_key: LayerwisePartialKey | None,
    store_pipeline_ranks: Mapping[int, tuple[int, ...]] | None,
):
    load: dict[int, Callable] = {}
    store: dict[int, Callable] = {}
    lookup: dict[int, Callable] = {}
    partial_keys: dict[int, Callable] = {}
    for group in topology.transfer_groups:
        metadata = group.key_metadata
        if layerwise_full_key is not None:
            load[group.group_id] = partial(
                _layerwise_local_keys,
                make_key=layerwise_full_key,
                group_id=group.group_id,
                head_rank=metadata.head_or_tp_rank,
                pp_rank=metadata.pp_rank,
            )
            store[group.group_id] = load[group.group_id]
            lookup[group.group_id] = partial(
                _layerwise_lookup_keys,
                make_key=layerwise_full_key,
                group_id=group.group_id,
                pp_size=topology.pp_size,
                head_rank_count=(topology.tp_size if group.uses_align_state else topology.tp_partition.key_rank_count),
            )
            if layerwise_partial_key is not None:
                partial_keys[group.group_id] = partial(
                    layerwise_partial_key,
                    group_id=group.group_id,
                    head_rank=metadata.head_or_tp_rank,
                    pp_rank=metadata.pp_rank,
                )
            continue

        load_prefixes = _local_prefixes(topology, group.group_id)
        store_prefixes = _store_prefixes(topology, group.group_id, load_prefixes, store_pipeline_ranks)
        lookup_prefixes = _lookup_prefixes(topology, group.group_id)
        load[group.group_id] = partial(_prefixed_keys, load_prefixes)
        store[group.group_id] = partial(_prefixed_keys, store_prefixes)
        lookup[group.group_id] = partial(_prefixed_keys, lookup_prefixes)

    return (
        partial(_dispatch, load),
        partial(_dispatch, store),
        partial(_dispatch, lookup),
        None if not partial_keys else partial(_dispatch, partial_keys),
    )


def _prefixed_keys(prefixes: tuple[str, ...], hashes: Sequence[BlockHash | str]) -> KeyAxes:
    hash_strings = tuple(block_hash_to_str(value) for value in hashes)
    return tuple(tuple(prefix + value for value in hash_strings) for prefix in prefixes)


def _layerwise_local_keys(
    hashes: Sequence[BlockHash | str], *, make_key: LayerwiseFullKey, group_id: int, head_rank: int, pp_rank: int
) -> KeyAxes:
    return (tuple(make_key(group_id, block_hash_to_str(value), head_rank, pp_rank) for value in hashes),)


def _layerwise_lookup_keys(
    hashes: Sequence[BlockHash | str], *, make_key: LayerwiseFullKey, group_id: int, pp_size: int, head_rank_count: int
) -> KeyAxes:
    hash_strings = tuple(block_hash_to_str(value) for value in hashes)
    return tuple(
        tuple(make_key(group_id, value, head_rank, pp_rank) for value in hash_strings)
        for pp_rank in range(pp_size)
        for head_rank in range(head_rank_count)
    )


def _local_prefixes(topology: KVPoolTopology, group_id: int) -> tuple[str, ...]:
    group = next(group for group in topology.transfer_groups if group.group_id == group_id)
    if not topology.tp_partition.tp_mismatch:
        return (_prefix(group.key_metadata),)
    return tuple(
        _prefix(
            replace(
                group.key_metadata,
                head_or_tp_rank=topology.tp_rank * topology.tp_partition.key_slices_per_rank + slice_index,
            )
        )
        for slice_index in range(topology.tp_partition.key_slices_per_rank)
    )


def _store_prefixes(
    topology: KVPoolTopology,
    group_id: int,
    load_prefixes: tuple[str, ...],
    store_pipeline_ranks: Mapping[int, tuple[int, ...]] | None,
) -> tuple[str, ...]:
    if store_pipeline_ranks is None:
        return load_prefixes
    group = next(group for group in topology.transfer_groups if group.group_id == group_id)
    return tuple(_prefix(replace(group.key_metadata, pp_rank=pp_rank)) for pp_rank in store_pipeline_ranks[group_id])


def _lookup_prefixes(topology: KVPoolTopology, group_id: int) -> tuple[str, ...]:
    group = next(group for group in topology.transfer_groups if group.group_id == group_id)
    rank_count = topology.tp_size if group.uses_align_state else topology.tp_partition.key_rank_count
    return tuple(
        _prefix(replace(group.key_metadata, pp_rank=pp_rank, dcp_rank=dcp_rank, head_or_tp_rank=head_rank))
        for pp_rank in range(topology.pp_size)
        for dcp_rank in range(topology.dcp_size)
        for head_rank in range(rank_count)
    )


def _prefix(metadata) -> str:
    return PoolKey(metadata, "").to_string()


# =============================================================================
# Group Dispatch and Row Values
# =============================================================================
#
# Route calls to the rule selected for each cache Group and construct the
# immutable row arrays and token-boundary hashes shared by the sections above.


def _dispatch(functions: dict[int, Callable], group_id: int, *args, **kwargs):
    return functions[group_id](*args, **kwargs)


def _empty_chunk_rows() -> ChunkRows:
    empty = _readonly(())
    return empty, empty, ()


def _empty_block_rows() -> BlockRows:
    empty = _readonly(())
    return empty, (), empty


def _readonly(values) -> ByteArray:
    result = np.fromiter(values, dtype=np.uint64)
    result.flags.writeable = False
    return result


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _boundary_hash(
    boundary_token: int, block_hashes: Sequence[BlockHash | str], hash_block_size: int
) -> BlockHash | str:
    if boundary_token <= 0 or boundary_token % hash_block_size:
        raise ValueError(f"Token boundary {boundary_token} is not aligned to the hash block size")
    hash_index = boundary_token // hash_block_size - 1
    if hash_index >= len(block_hashes):
        raise ValueError(f"Token boundary {boundary_token} has no corresponding block hash")
    return block_hashes[hash_index]
