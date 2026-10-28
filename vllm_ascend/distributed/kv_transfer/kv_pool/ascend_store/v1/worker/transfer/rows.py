"""Materialize request-time rows and select their Worker ownership."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_block_hashes

from ...infrastructure.coordinator import store_block_range

if TYPE_CHECKING:
    from .batch import KeysByPartition

UInt64Array: TypeAlias = NDArray[np.uint64]
LookupChunkRanges: TypeAlias = tuple[UInt64Array, UInt64Array, tuple[BlockHash | str, ...]]
TransferRecords: TypeAlias = tuple[UInt64Array, tuple[BlockHash | str, ...], UInt64Array]
StoreTransferRecords: TypeAlias = tuple[list[int], list[BlockHash | str], list[int]]


def lookup_chunk_ranges(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
) -> LookupChunkRanges:
    """Map the requested token interval to ordered external lookup objects."""

    if not block_hashes or end_token <= 0:
        return _empty_lookup_chunk_ranges()
    grouped_hashes = get_block_hashes(block_hashes, block_size, hash_block_size)
    logical_count = min(len(grouped_hashes), _ceil_div(end_token, block_size))
    first_block = max(start_token // block_size, 0)
    starts: list[int] = []
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    for block_index in range(first_block, logical_count):
        start = block_index * block_size
        end = min(start + block_size, end_token)
        if mask is not None and (block_index >= len(mask) or not mask[block_index]):
            continue
        starts.append(start)
        counts.append(end - start)
        hashes.append(grouped_hashes[block_index])
    return _readonly(starts), _readonly(counts), tuple(hashes)


def fine_lookup_chunk_ranges(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
) -> LookupChunkRanges:
    """Expose each completed hash boundary while retaining its cache-block identity."""

    if not block_hashes or end_token <= 0:
        return _empty_lookup_chunk_ranges()
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


def load_transfer_records(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    local_block_ids: Sequence[int],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    tail_boundary_token: int | None = None,
    mask: Sequence[bool] | None = None,
    minimum_block_id: int = 0,
) -> TransferRecords:
    """Select local block rows for one ordinary, TP-mismatch, or Consumer-PP Load."""

    if not block_hashes or end_token <= 0:
        return _empty_transfer_records()
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
    block_offset = max(logical_count - len(local_block_ids), 0)
    first_block = max(start_token // block_size, block_offset, 0)
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    selected_block_ids: list[int] = []
    for block_index in range(first_block, logical_count):
        start = block_index * block_size
        end = min(start + block_size, end_token)
        if mask is not None and (block_index >= len(mask) or not mask[block_index]):
            continue
        local_index = block_index - block_offset
        block_id = local_block_ids[local_index]
        if block_id < minimum_block_id:
            continue
        if tail_block_index is not None and block_index == tail_block_index:
            assert tail_boundary_token is not None
            hashes.append(boundary_hash(tail_boundary_token, block_hashes, hash_block_size))
        else:
            if block_index >= len(grouped_hashes):
                continue
            hashes.append(grouped_hashes[block_index])
        counts.append(end - start)
        selected_block_ids.append(block_id)
    return _readonly(counts), tuple(hashes), _readonly(selected_block_ids)


def store_transfer_records(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    local_block_ids: Sequence[int],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
    minimum_block_id: int = 0,
) -> StoreTransferRecords:
    """Select Store rows; mask positions start at the block containing start_token."""

    if not block_hashes or end_token <= 0:
        return [], [], []
    grouped_hashes = get_block_hashes(block_hashes, block_size, hash_block_size)
    indices = store_block_range(start_token, end_token, block_size, len(grouped_hashes), len(local_block_ids))
    block_offset = max(indices.stop - len(local_block_ids), 0)
    mask_start_block = max(start_token // block_size, 0)
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    selected_block_ids: list[int] = []
    for block_index in indices:
        start = block_index * block_size
        end = min(start + block_size, end_token)
        local_index = block_index - block_offset
        mask_index = block_index - mask_start_block
        if mask is not None and (mask_index >= len(mask) or not mask[mask_index]):
            continue
        block_id = local_block_ids[local_index]
        if block_id < minimum_block_id:
            continue
        counts.append(end - start)
        hashes.append(grouped_hashes[block_index])
        selected_block_ids.append(block_id)
    return counts, hashes, selected_block_ids


def _empty_lookup_chunk_ranges() -> LookupChunkRanges:
    empty = _readonly(())
    return empty, empty, ()


def _empty_transfer_records() -> TransferRecords:
    empty = _readonly(())
    return empty, (), empty


def _readonly(values) -> UInt64Array:
    result = np.fromiter(values, dtype=np.uint64)
    result.flags.writeable = False
    return result


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def boundary_hash(
    boundary_token: int,
    block_hashes: Sequence[BlockHash | str],
    hash_block_size: int,
) -> BlockHash | str:
    """Return the content hash that names one exact token boundary."""

    if boundary_token <= 0 or boundary_token % hash_block_size:
        raise ValueError(f"Token boundary {boundary_token} is not aligned to the hash block size")
    hash_index = boundary_token // hash_block_size - 1
    if hash_index >= len(block_hashes):
        raise ValueError(f"Token boundary {boundary_token} has no corresponding block hash")
    return block_hashes[hash_index]


def merge_keys_by_partition(group_id: int, key_parts: list[KeysByPartition]) -> KeysByPartition:
    """Concatenate per-request key coordinates without changing axes."""

    partition_count = len(key_parts[0]) if key_parts else 0
    if any(len(keys) != partition_count for keys in key_parts):
        raise RuntimeError(f"Cache group {group_id} changed its key coordinate count at runtime")
    return tuple(
        tuple(key for keys in key_parts for key in keys[partition_index]) for partition_index in range(partition_count)
    )


def split_selection(selection: tuple[bool, ...], partition_count: int) -> tuple[tuple[bool, ...], ...]:
    """Restore flattened object admission to key-axis coordinates."""

    if partition_count == 0:
        return ()
    if len(selection) % partition_count:
        raise RuntimeError("Store selection does not form aligned key axes")
    transfer_record_count = len(selection) // partition_count
    return tuple(
        selection[index * transfer_record_count : (index + 1) * transfer_record_count]
        for index in range(partition_count)
    )


def compact_single_partition_records(
    transfer_records_by_request: tuple[StoreTransferRecords, ...],
    partition_keys: tuple[str, ...],
    selected_transfer_record_indices: tuple[bool, ...],
) -> tuple[list[int], list[int], tuple[str, ...]]:
    """Compact a one-axis Store batch while preserving request splits."""

    local_block_ids: list[int] = []
    selected_keys: list[str] = []
    request_transfer_record_boundaries = [0]
    transfer_record_start = 0
    for counts, _hashes, ids in transfer_records_by_request:
        request_transfer_record_count = len(counts)
        selection = selected_transfer_record_indices[
            transfer_record_start : transfer_record_start + request_transfer_record_count
        ]
        keys = partition_keys[transfer_record_start : transfer_record_start + request_transfer_record_count]
        for block_id, key, keep in zip(ids, keys, selection, strict=True):
            if keep:
                local_block_ids.append(block_id)
                selected_keys.append(key)
        request_transfer_record_boundaries.append(len(local_block_ids))
        transfer_record_start += request_transfer_record_count
    return local_block_ids, request_transfer_record_boundaries, tuple(selected_keys)


def empty_store_transfer_records() -> StoreTransferRecords:
    return [], [], []


def empty_rows() -> TransferRecords:
    empty: np.ndarray = np.empty(0, dtype=np.uint64)
    empty.flags.writeable = False
    return empty, (), empty


def concat_uint64(parts: Sequence[np.ndarray]) -> np.ndarray:
    if not parts:
        result: np.ndarray = np.empty(0, dtype=np.uint64)
    elif len(parts) == 1:
        result = parts[0]
    else:
        result = np.concatenate(parts)
        result.flags.writeable = False
    return result


def readonly_indices(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.intp)
    result.flags.writeable = False
    return result


def readonly_uint64(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.uint64)
    result.flags.writeable = False
    return result


def readonly_bool(values: Sequence[bool]) -> np.ndarray:
    result = np.asarray(values, dtype=np.bool_)
    result.flags.writeable = False
    return result
