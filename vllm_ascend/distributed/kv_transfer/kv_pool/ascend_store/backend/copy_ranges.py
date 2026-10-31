# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GVA packet execution shared by ordinary pooling and pooled PD.

Allocation, leases, completion events and publication belong to the caller.
Legacy limits count input addresses and bytes per packet. A caller can also
bound aggregate bytes per native submission without changing those defaults.
"""

from collections.abc import Callable

import numpy as np

RangeBatch = tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]


def split_transfer_packets(gvas, addrs, sizes, max_transfer_bytes):
    if max_transfer_bytes <= 0 or not len(sizes):
        return gvas, addrs, sizes
    split_counts = (sizes + max_transfer_bytes - 1) // max_transfer_bytes
    if int(split_counts.sum()) == sizes.shape[0]:
        return gvas, addrs, sizes
    split_indices = np.arange(int(split_counts.max()), dtype=np.int64)
    split_mask = split_indices[:, None] < split_counts[None, :]
    entry_indices = np.broadcast_to(np.arange(sizes.shape[0], dtype=np.int64), split_mask.shape)[split_mask]
    offsets = np.broadcast_to(split_indices[:, None] * max_transfer_bytes, split_mask.shape)[split_mask]
    return (
        gvas[entry_indices] + offsets,
        addrs[entry_indices] + offsets,
        np.minimum(max_transfer_bytes, sizes[entry_indices] - offsets),
    )


def copy_gva_ranges(
    copy: Callable[[list[int], list[int], list[int], int], int],
    gvas,
    addrs,
    sizes,
    direction: int,
    *,
    max_addresses: int = 0,
    max_packet_bytes: int = 0,
    max_batch_bytes: int = 0,
    cancelled: Callable[[], bool] | None = None,
) -> int | None:
    """Copy vectors until failure or cancellation, draining each native call.

    Cancellation returns None only between submissions. The caller still owns
    the buffers until this function returns; an in-flight call is never cut
    short, and a cancelled partial copy must not be published as complete.
    """
    gvas, addrs, sizes = (np.asarray(values, dtype=np.int64) for values in (gvas, addrs, sizes))
    if any(values.ndim != 1 for values in (gvas, addrs, sizes)) or not (len(gvas) == len(addrs) == len(sizes)):
        raise ValueError("Pool copy vectors must be aligned one-dimensional arrays")
    if not len(gvas):
        return 0
    if np.any(gvas <= 0) or np.any(addrs <= 0) or np.any(sizes <= 0):
        raise ValueError("Pool copy addresses and sizes must be positive")
    if not len(gvas):
        return 0
    input_limit = max_addresses if max_addresses > 0 else len(gvas)
    packet_limit = max_packet_bytes
    if max_batch_bytes > 0:
        packet_limit = min(packet_limit, max_batch_bytes) if packet_limit > 0 else max_batch_bytes
    for start in range(0, len(gvas), input_limit):
        stop = start + input_limit
        addresses, buffers, lengths = split_transfer_packets(
            gvas[start:stop], addrs[start:stop], sizes[start:stop], packet_limit
        )
        if max_batch_bytes <= 0:
            if cancelled is not None and cancelled():
                return None
            result = copy(addresses.tolist(), buffers.tolist(), lengths.tolist(), direction)
            if result != 0:
                return result
            if cancelled is not None and cancelled():
                return None
            continue
        first = total = 0
        for index in range(len(lengths) + 1):
            if index == len(lengths) or (
                index > first and (total + int(lengths[index]) > max_batch_bytes or index - first >= input_limit)
            ):
                if cancelled is not None and cancelled():
                    return None
                result = copy(
                    addresses[first:index].tolist(),
                    buffers[first:index].tolist(),
                    lengths[first:index].tolist(),
                    direction,
                )
                if result != 0:
                    return result
                if cancelled is not None and cancelled():
                    return None
                first, total = index, 0
            if index < len(lengths):
                total += int(lengths[index])
    return 0


def range_transfer_batches(
    keys: list[str],
    all_buffers: list[list[int]],
    all_sizes: list[list[int]],
    all_offsets: list[list[int]],
    max_transfer_blocks: int,
    max_transfer_bytes: int,
    *,
    max_batch_bytes: int = 0,
    max_batch_ranges: int = 0,
) -> list[RangeBatch]:
    """Split backend range calls by key count and per-segment byte size."""
    if max_batch_bytes > 0:
        max_transfer_bytes = min(max_transfer_bytes, max_batch_bytes) if max_transfer_bytes > 0 else max_batch_bytes
    row_count = len(keys)
    if not (len(all_buffers) == len(all_sizes) == len(all_offsets) == row_count):
        raise ValueError("Layerwise range metadata must contain one buffer/size/offset row per key")

    # Default configuration (both limits unset) means one batch holding every
    # key and no per-segment splitting, so every row passes through as-is
    # unless it carries an empty or negative segment. Detect that case and
    # hand back the caller's own lists: this runs once per key, per layer,
    # per group, and rebuilding them element-wise was the receive thread's
    # largest single block of exposed time.
    if max_transfer_blocks <= 0 and max_transfer_bytes <= 0:
        for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
            if not (len(buffers) == len(sizes) == len(offsets)):
                raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
            if sizes and min(sizes) >= 0 and 0 not in sizes:
                continue
            break
        else:
            return _bound_range_batches(
                [(keys, all_buffers, all_sizes, all_offsets)], max_batch_bytes, max_batch_ranges
            )

    normalized_buffers: list[list[int]] = []
    normalized_sizes: list[list[int]] = []
    normalized_offsets: list[list[int]] = []
    for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
        if not (len(buffers) == len(sizes) == len(offsets)):
            raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
        split_buffers: list[int] = []
        split_sizes: list[int] = []
        split_offsets: list[int] = []
        for buffer, size, offset in zip(buffers, sizes, offsets, strict=True):
            if size < 0:
                raise ValueError(f"Layerwise range size must be non-negative, got {size}")
            if size == 0:
                continue
            packet_size = max_transfer_bytes if max_transfer_bytes > 0 else size
            for packet_offset in range(0, size, packet_size):
                split_buffers.append(buffer + packet_offset)
                split_sizes.append(min(packet_size, size - packet_offset))
                split_offsets.append(offset + packet_offset)
        normalized_buffers.append(split_buffers)
        normalized_sizes.append(split_sizes)
        normalized_offsets.append(split_offsets)

    rows_per_batch = max_transfer_blocks if max_transfer_blocks > 0 else max(1, row_count)
    batches = [
        (
            keys[start:end],
            normalized_buffers[start:end],
            normalized_sizes[start:end],
            normalized_offsets[start:end],
        )
        for start in range(0, row_count, rows_per_batch)
        for end in [min(start + rows_per_batch, row_count)]
    ]
    return _bound_range_batches(batches, max_batch_bytes, max_batch_ranges)


def _bound_range_batches(batches: list[RangeBatch], max_bytes: int, max_ranges: int) -> list[RangeBatch]:
    if max_bytes <= 0 and max_ranges <= 0:
        return batches
    result = []
    for keys, buffers, sizes, offsets in batches:
        current: RangeBatch = ([], [], [], [])
        total = count = 0
        active_row = -1
        for row, key in enumerate(keys):
            for buffer, size, offset in zip(buffers[row], sizes[row], offsets[row], strict=True):
                if count and ((max_bytes > 0 and total + size > max_bytes) or (max_ranges > 0 and count >= max_ranges)):
                    result.append(current)
                    current = ([], [], [], [])
                    total = count = 0
                    active_row = -1
                if active_row != row:
                    current[0].append(key)
                    for values in current[1:]:
                        values.append([])
                    active_row = row
                current[1][-1].append(buffer)
                current[2][-1].append(size)
                current[3][-1].append(offset)
                total += size
                count += 1
        if count:
            result.append(current)
    return result


def copy_key_ranges(
    copy: Callable[[list[str], list[list[int]], list[list[int]], list[list[int]]], list[int]],
    keys: list[str],
    buffers: list[list[int]],
    sizes: list[list[int]],
    offsets: list[list[int]],
    *,
    max_ranges: int,
    max_bytes: int,
    operation: str,
) -> None:
    """Execute bounded key-range copies; a failed batch ends this restore/write.

    Range APIs return either zero or a byte count on success, as in the
    ordinary pool sender/receiver. Session start/commit status codes have a
    different contract and must still be checked for equality with zero.
    """
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import require_aligned_batch_results

    for batch_keys, batch_buffers, batch_sizes, batch_offsets in range_transfer_batches(
        keys, buffers, sizes, offsets, max_ranges, max_bytes, max_batch_bytes=max_bytes, max_batch_ranges=max_ranges
    ):
        codes = require_aligned_batch_results(
            operation, batch_keys, copy(batch_keys, batch_buffers, batch_sizes, batch_offsets)
        )
        if any(code < 0 for code in codes):
            raise RuntimeError(f"{operation} failed: {codes}")
