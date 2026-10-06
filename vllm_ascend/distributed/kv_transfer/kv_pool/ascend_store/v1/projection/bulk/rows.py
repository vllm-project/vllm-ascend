"""Project request-time Bulk rows and select their writer ownership."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_block_hashes

from ...topology import KVPoolTopology

ByteArray: TypeAlias = NDArray[np.uint64]
ChunkRows: TypeAlias = tuple[ByteArray, ByteArray, tuple[BlockHash | str, ...]]
BlockRows: TypeAlias = tuple[ByteArray, tuple[BlockHash | str, ...], ByteArray]
StoreCandidateRows: TypeAlias = tuple[list[int], list[BlockHash | str], list[int]]


def lookup_chunk_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
) -> ChunkRows:
    """Map the requested token interval to ordered external lookup objects."""

    if not block_hashes or end_token <= 0:
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


def fine_lookup_chunk_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
) -> ChunkRows:
    """Expose each completed hash boundary while retaining its cache-block identity."""

    if not block_hashes or end_token <= 0:
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


def load_block_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    block_ids: Sequence[int],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    tail_boundary_token: int | None = None,
    mask: Sequence[bool] | None = None,
    minimum_block_id: int = 0,
) -> BlockRows:
    """Select local block rows for one ordinary, TP-mismatch, or Consumer-PP Load."""

    if not block_hashes or end_token <= 0:
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


def store_candidate_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    block_ids: Sequence[int],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
    minimum_block_id: int = 0,
) -> StoreCandidateRows:
    """Select Store rows before Backend admission or writer partitioning."""

    if not block_hashes or end_token <= 0:
        return [], [], []
    grouped_hashes = get_block_hashes(block_hashes, block_size, hash_block_size)
    logical_count = min(len(grouped_hashes), _ceil_div(end_token, block_size))
    block_offset = max(logical_count - len(block_ids), 0)
    aligned_start_token = start_token // block_size * block_size
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    selected_block_ids: list[int] = []
    for block_index in range(logical_count):
        start = block_index * block_size
        end = min(start + block_size, end_token)
        local_index = block_index - block_offset
        if start < aligned_start_token or end <= start or local_index < 0 or local_index >= len(block_ids):
            continue
        if mask is not None and (block_index >= len(mask) or not mask[block_index]):
            continue
        block_id = block_ids[local_index]
        if block_id < minimum_block_id:
            continue
        counts.append(end - start)
        hashes.append(grouped_hashes[block_index])
        selected_block_ids.append(block_id)
    return counts, hashes, selected_block_ids


def select_store_writer_rows(
    topology: KVPoolTopology,
    rows: StoreCandidateRows,
    *,
    tp_mismatch: bool,
    align_state: bool = False,
) -> StoreCandidateRows:
    """Partition rows only among ranks proven to publish the same key and bytes."""

    if tp_mismatch or align_state:
        replica_count = 1
    elif topology.dcp_size > 1:
        # BulkProjectionBinder rejects topologies where one (head, dcp) key still has
        # multiple writers or where DCP spans PCP. Every remaining local key is
        # therefore unique and must publish the complete row set.
        replica_count = 1
    else:
        replica_count = topology.put_step
    shard_rank = topology.pcp_rank * replica_count + topology.tp_rank % replica_count
    shard_count = topology.pcp_size * replica_count
    if shard_count <= 1:
        return rows
    counts, hashes, block_ids = rows
    selection = slice(shard_rank, len(counts), shard_count)
    return counts[selection], hashes[selection], block_ids[selection]


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
    boundary_token: int,
    block_hashes: Sequence[BlockHash | str],
    hash_block_size: int,
) -> BlockHash | str:
    if boundary_token <= 0 or boundary_token % hash_block_size:
        raise ValueError(f"Token boundary {boundary_token} is not aligned to the hash block size")
    hash_index = boundary_token // hash_block_size - 1
    if hash_index >= len(block_hashes):
        raise ValueError(f"Token boundary {boundary_token} has no corresponding block hash")
    return block_hashes[hash_index]


def boundary_hash(
    boundary_token: int,
    block_hashes: Sequence[BlockHash | str],
    hash_block_size: int,
) -> BlockHash | str:
    """Return the content hash that names one exact token boundary."""

    return _boundary_hash(boundary_token, block_hashes, hash_block_size)
