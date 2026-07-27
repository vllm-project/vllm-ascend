#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Transparent object chunking for Mooncake's local SSD path.

Mooncake stages each SSD object in a contiguous CacheLib allocation. Hybrid
models can aggregate multiple layers into one logical KV object that is larger
than a CacheLib slab, so keep scheduler-facing keys logical and split only the
backend representation.
"""

from collections.abc import Iterable

DEFAULT_SSD_OBJECT_CHUNK_BYTES = 3 * 1024 * 1024
DEFAULT_SSD_READ_BATCH_BYTES = 512 * 1024 * 1024
DEFAULT_SSD_READ_BATCH_OBJECTS = 256
SSD_CHUNK_KEY_SUFFIX = "@mcssd1:"


def _split_ssd_object(
    key: str,
    addrs: list[int],
    sizes: list[int],
    max_chunk_bytes: int,
) -> list[tuple[str, list[int], list[int]]]:
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    if len(addrs) != len(sizes):
        raise ValueError("Mooncake buffer address and size counts differ")

    chunks: list[tuple[list[int], list[int]]] = []
    chunk_addrs: list[int] = []
    chunk_sizes: list[int] = []
    chunk_bytes = 0

    def flush() -> None:
        nonlocal chunk_addrs, chunk_sizes, chunk_bytes
        if chunk_sizes:
            chunks.append((chunk_addrs, chunk_sizes))
            chunk_addrs = []
            chunk_sizes = []
            chunk_bytes = 0

    for raw_addr, raw_size in zip(addrs, sizes, strict=True):
        addr = int(raw_addr)
        size = int(raw_size)
        if size < 0:
            raise ValueError("Mooncake buffer size must not be negative")
        offset = 0
        while offset < size:
            if chunk_bytes == max_chunk_bytes:
                flush()
            take = min(size - offset, max_chunk_bytes - chunk_bytes)
            chunk_addrs.append(addr + offset)
            chunk_sizes.append(take)
            chunk_bytes += take
            offset += take
        if chunk_bytes == max_chunk_bytes:
            flush()

    flush()
    if not chunks:
        chunks.append(([], []))
    return [
        (
            f"{key}{SSD_CHUNK_KEY_SUFFIX}{index}",
            chunk_addrs,
            chunk_sizes,
        )
        for index, (chunk_addrs, chunk_sizes) in enumerate(chunks)
    ]


def split_ssd_batch(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    max_chunk_bytes: int = DEFAULT_SSD_OBJECT_CHUNK_BYTES,
) -> tuple[list[str], list[list[int]], list[list[int]], list[tuple[int, int]]]:
    """Expand logical objects into deterministic SSD-sized objects."""
    if not (len(keys) == len(addrs) == len(sizes)):
        raise ValueError("Mooncake key, address, and size counts differ")

    expanded_keys: list[str] = []
    expanded_addrs: list[list[int]] = []
    expanded_sizes: list[list[int]] = []
    result_ranges: list[tuple[int, int]] = []

    for key, object_addrs, object_sizes in zip(keys, addrs, sizes, strict=True):
        start = len(expanded_keys)
        for chunk_key, chunk_addrs, chunk_sizes in _split_ssd_object(
            key,
            object_addrs,
            object_sizes,
            max_chunk_bytes,
        ):
            expanded_keys.append(chunk_key)
            expanded_addrs.append(chunk_addrs)
            expanded_sizes.append(chunk_sizes)
        result_ranges.append((start, len(expanded_keys)))

    return expanded_keys, expanded_addrs, expanded_sizes, result_ranges


def ssd_chunk_head_keys(keys: list[str]) -> list[str]:
    """Return logical existence markers for SSD-chunked objects."""
    return [f"{key}{SSD_CHUNK_KEY_SUFFIX}0" for key in keys]


def iter_ssd_read_batches(
    sizes: list[list[int]],
    max_batch_bytes: int,
    max_batch_objects: int = DEFAULT_SSD_READ_BATCH_OBJECTS,
) -> Iterable[tuple[int, int]]:
    """Bound each native BatchGet below the registered staging buffer."""
    if max_batch_bytes <= 0 or max_batch_objects <= 0:
        raise ValueError("SSD read batch limits must be positive")

    start = 0
    batch_bytes = 0
    for index, object_sizes in enumerate(sizes):
        object_bytes = sum(int(size) for size in object_sizes)
        if object_bytes > max_batch_bytes:
            raise ValueError(
                f"SSD object chunk ({object_bytes} bytes) exceeds read batch limit ({max_batch_bytes} bytes)"
            )
        if index > start and (batch_bytes + object_bytes > max_batch_bytes or index - start >= max_batch_objects):
            yield start, index
            start = index
            batch_bytes = 0
        batch_bytes += object_bytes

    if start < len(sizes):
        yield start, len(sizes)


def aggregate_chunk_results(
    expanded_results: list[int],
    result_ranges: list[tuple[int, int]],
) -> list[int]:
    """Collapse physical chunk statuses to one status per logical object."""
    logical_results: list[int] = []
    for start, end in result_ranges:
        if start < 0 or end <= start or end > len(expanded_results):
            raise ValueError("Invalid Mooncake SSD chunk result range")
        chunk_results = expanded_results[start:end]
        first_failure = next(
            (int(value) for value in chunk_results if value < 0),
            None,
        )
        logical_results.append(first_failure if first_failure is not None else 0)
    return logical_results
