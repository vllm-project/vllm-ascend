# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Iterator
from typing import Any

_SPLIT_ALIGNMENT = 256


class KVReadBatcher:
    """Bound KV read bursts, with FIFO admission per receiver and per batch.

    A zero byte limit preserves the original transfer call and concurrency.
    This does not wrap collectives or impose a link-wide bandwidth limit.
    """

    def __init__(self, max_bytes: int = 0, max_descriptors: int = 128):
        if type(max_bytes) is not int or max_bytes < 0 or max_bytes % _SPLIT_ALIGNMENT:
            raise ValueError("kv_read_batch_bytes must be zero or a positive multiple of 256")
        if type(max_descriptors) is not int or max_descriptors < 1:
            raise ValueError("kv_read_batch_descriptors must be a positive integer")
        self.max_bytes = max_bytes
        self.max_descriptors = max_descriptors
        self._condition = threading.Condition()
        self._issued = 0
        self._serving = 0

    def _batches(
        self, sources: list[int], destinations: list[int], sizes: list[int]
    ) -> Iterator[tuple[list[int], list[int], list[int]]]:
        if len(sources) != len(destinations) or len(sources) != len(sizes):
            raise ValueError("KV read descriptor counts differ")
        if any(size < 0 for size in sizes):
            raise ValueError("KV read sizes must be nonnegative")
        batch_src, batch_dst, batch_sizes = [], [], []
        used = 0
        for source, destination, size in zip(sources, destinations, sizes):
            source, destination, size = int(source), int(destination), int(size)
            offset = 0
            while offset < size or (size == 0 and offset == 0):
                space = self.max_bytes - used
                remaining = size - offset
                take = remaining if remaining <= space else space // _SPLIT_ALIGNMENT * _SPLIT_ALIGNMENT
                if len(batch_sizes) >= self.max_descriptors or (remaining and take == 0):
                    yield batch_src, batch_dst, batch_sizes
                    batch_src, batch_dst, batch_sizes, used = [], [], [], 0
                    continue
                batch_src.append(source + offset)
                batch_dst.append(destination + offset)
                batch_sizes.append(take)
                offset += take
                used += take
                if used == self.max_bytes:
                    yield batch_src, batch_dst, batch_sizes
                    batch_src, batch_dst, batch_sizes, used = [], [], [], 0
                if size == 0:
                    break
        if batch_sizes:
            yield batch_src, batch_dst, batch_sizes

    def read(self, engine: Any, session: str, sources: list[int], destinations: list[int], sizes: list[int]) -> int:
        if not self.max_bytes:
            return engine.batch_transfer_sync_read(session, sources, destinations, sizes)
        for src, dst, lengths in self._batches(sources, destinations, sizes):
            with self._condition:
                ticket = self._issued
                self._issued += 1
                while ticket != self._serving:
                    self._condition.wait()
            try:
                result = engine.batch_transfer_sync_read(session, src, dst, lengths)
                if result < 0:
                    return result
            finally:
                with self._condition:
                    self._serving += 1
                    self._condition.notify_all()
        return 0
