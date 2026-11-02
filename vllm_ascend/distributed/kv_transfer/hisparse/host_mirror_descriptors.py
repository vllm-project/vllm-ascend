# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Validate and snapshot CPU descriptors for mapped HiSparse Host writes."""

import numpy as np

MAX_I64 = (1 << 63) - 1


def prepare_snapshot(
    source, destination, sizes, device_extents, host_extents, max_aiv_bytes=10496, min_aiv_descriptors=1
):
    """Return a private 3xN snapshot with real mapped aliases and eligibility.

    Each device extent is (base, bytes); each Host extent is
    (CPU base, bytes, mapped device alias). No device values are inspected.
    Overlapping destinations keep the original ordered DMA implementation.
    """
    arrays = [np.asarray(x) for x in (source, destination, sizes)]
    if any(x.ndim != 1 or x.dtype != np.dtype("int64") for x in arrays):
        raise ValueError("Descriptor vectors must be CPU int64 arrays")
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("Descriptor vector shapes differ")
    count = arrays[0].size
    if count > 1 << 20:
        raise ValueError("Descriptor count is unbounded")
    snapshot = np.stack(arrays).copy()
    if count == 0:
        return snapshot, False, "empty"
    src, dst, length = snapshot
    if (
        np.any(src <= 0)
        or np.any(dst <= 0)
        or np.any(length <= 0)
        or np.any(src > MAX_I64 - length)
        or np.any(dst > MAX_I64 - length)
    ):
        raise ValueError("Invalid or overflowing descriptor address/length")
    device_matches = np.zeros(count, dtype=np.int16)
    for base, size in device_extents:
        if not 0 < base <= MAX_I64 or not 0 < size <= MAX_I64 - base:
            raise ValueError("Invalid retained device extent")
        device_matches += (src >= base) & (src + length <= base + size)
    if np.any(device_matches != 1):
        raise ValueError("Source must belong to one retained device allocation")
    host_matches = np.zeros(count, dtype=np.int16)
    aliases = np.empty(count, dtype=np.int64)
    for base, size, alias in host_extents:
        if not 0 < base <= MAX_I64 or not 0 < alias <= MAX_I64 or not 0 < size <= min(MAX_I64 - base, MAX_I64 - alias):
            raise ValueError("Invalid retained Host mapping extent")
        mask = (dst >= base) & (dst + length <= base + size)
        host_matches += mask
        aliases[mask] = alias + (dst[mask] - base)
    if np.any(host_matches != 1):
        raise ValueError("Destination must belong to one live Host registration")
    order = np.argsort(dst, kind="stable")
    overlap = np.any(dst[order][1:] < (dst + length)[order][:-1])
    snapshot[1] = aliases
    if overlap:
        return snapshot, False, "ordered_overlapping_destinations"
    if np.any(length > max_aiv_bytes):
        return snapshot, False, "large_extent_keeps_native_DMA"
    if count < min_aiv_descriptors:
        return snapshot, False, "small_batch_keeps_native_DMA"
    return snapshot, True, "small_disjoint_extents"
