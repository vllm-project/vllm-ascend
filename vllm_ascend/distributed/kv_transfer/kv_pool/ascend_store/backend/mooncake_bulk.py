# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded DDR staging for HiSparse Host Source pages in Mooncake Store."""

from __future__ import annotations

import ctypes
import mmap
import threading
from bisect import bisect_right
from collections.abc import Iterator
from typing import Any

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import require_aligned_batch_results

# Each direction owns a bounded buffer; network waits never serialize PUT and GET.
_HOST_STAGING_BYTES = 64 * 1024 * 1024


class _HostStaging:
    def __init__(self) -> None:
        self.buffer = mmap.mmap(-1, _HOST_STAGING_BYTES)
        self.base = ctypes.addressof(ctypes.c_ubyte.from_buffer(self.buffer))
        self.lock = threading.Lock()

    def prepare(self) -> None:
        # Fault pages in once at registration, outside the transfer hot path.
        # Store/HIXL owns transport registration; do not double-register with ACL.
        ctypes.memset(self.base, 0, _HOST_STAGING_BYTES)


class MooncakeMixedBulk:
    def __init__(self, store: Any, host_ranges: dict[int, int]) -> None:
        self.store = store
        self._host_ranges = host_ranges.copy()
        self._host_bases = sorted(host_ranges)
        self._registered: dict[int, int] = {}
        self._send = _HostStaging()
        try:
            self._recv = _HostStaging()
        except Exception:
            self._send.buffer.close()
            raise

    def register(self, ptrs: list[int], lengths: list[int]) -> None:
        if len(ptrs) != len(lengths):
            raise ValueError("Mooncake bulk buffer pointers and lengths must have the same size")
        regions = dict(zip(ptrs, lengths, strict=True))
        if not self._host_ranges or any(regions.get(ptr) != size for ptr, size in self._host_ranges.items()):
            raise ValueError("Mooncake bulk transfer requires explicit Host Source buffer ranges")
        if len(regions) != len(ptrs) or any(ptr <= 0 or size <= 0 for ptr, size in regions.items()):
            raise ValueError("Mooncake bulk buffer pointers must be distinct and sizes positive")
        # HIXL cannot register HiSparse's already mapped/pinned pages (Resource_Busy).
        # Use ordinary DDR for Store and keep the Indexer slab on the direct device path.
        for staging in (self._send, self._recv):
            staging.prepare()
            self._register(staging.base, _HOST_STAGING_BYTES)
        for ptr, size in regions.items():
            if ptr not in self._host_ranges:
                self._register(ptr, size)

    def _register(self, ptr: int, size: int) -> None:
        rc = self.store.register_buffer(ptr, size)
        if rc != 0:
            raise RuntimeError(f"Mooncake bulk buffer registration failed: {size} bytes, status={rc}")
        self._registered[ptr] = size

    def close(self) -> None:
        # Worker shutdown stops the transfer threads before releasing this buffer.
        failures = []
        for ptr in reversed(tuple(self._registered)):
            try:
                rc = self.store.unregister_buffer(ptr)
                if rc != 0:
                    logger.error("Mooncake bulk buffer unregister failed: status=%s", rc)
                    failures.append(ptr)
                else:
                    del self._registered[ptr]
            except Exception:
                logger.exception("Mooncake bulk buffer unregister raised")
                failures.append(ptr)
        if failures:
            # Keep the buffers alive until their owner has closed the Store.
            raise RuntimeError(f"Mooncake could not unregister {len(failures)} bulk buffers")
        self.release_staging()

    def release_staging(self) -> None:
        """Release memory after successful deregistration or backend Store shutdown."""
        self._registered.clear()
        self._recv.buffer.close()
        self._send.buffer.close()

    def _validate_host_key(self, addrs: list[int], sizes: list[int]) -> int:
        if not addrs or len(addrs) != len(sizes):
            raise ValueError("invalid Mooncake Host Source key geometry")
        for addr, size in zip(addrs, sizes, strict=True):
            index = bisect_right(self._host_bases, addr) - 1
            if index < 0 or size <= 0:
                raise ValueError("Mooncake Host Source address is outside its transfer buffer")
            base = self._host_bases[index]
            if addr + size > base + self._host_ranges[base]:
                raise ValueError("Mooncake Host Source address is outside its transfer buffer")
        key_bytes = sum(sizes)
        if key_bytes > _HOST_STAGING_BYTES:
            raise ValueError(f"Mooncake Host Source key exceeds {_HOST_STAGING_BYTES} byte staging capacity")
        return key_bytes

    def _host_batches(
        self, addrs: list[list[int]], sizes: list[list[int]], staging: _HostStaging
    ) -> Iterator[tuple[int, int, list[list[int]]]]:
        # Validate all ranges before any copy; addresses are raw pointers.
        totals = [self._validate_host_key(a, s) for a, s in zip(addrs, sizes, strict=True)]
        begin = 0
        cursor = staging.base
        staged: list[list[int]] = []
        for index, (parts, total) in enumerate(zip(sizes, totals, strict=True)):
            if cursor + total > staging.base + _HOST_STAGING_BYTES:
                yield begin, index, staged
                begin, cursor, staged = index, staging.base, []
            spans = []
            for size in parts:
                spans.append(cursor)
                cursor += size
            staged.append(spans)
        if staged:
            yield begin, len(sizes), staged

    def put(
        self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]], config: Any, *, host: bool
    ) -> list[bool]:
        if not len(keys) == len(addrs) == len(sizes):
            raise ValueError("Mooncake bulk PUT requires one buffer list per key")
        if not host:
            return self._put(keys, addrs, sizes, config)
        result = []
        with self._send.lock:
            for begin, end, staged in self._host_batches(addrs, sizes, self._send):
                for src, dst, lengths in zip(addrs[begin:end], staged, sizes[begin:end], strict=True):
                    for source, target, length in zip(src, dst, lengths, strict=True):
                        ctypes.memmove(target, source, length)
                result.extend(self._put(keys[begin:end], staged, sizes[begin:end], config))
        return result

    def _put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]], config: Any) -> list[bool]:
        codes = require_aligned_batch_results(
            "Mooncake bulk PUT", keys, self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
        )
        return [rc == 0 for rc in codes]

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]], *, host: bool) -> list[int]:
        if not len(keys) == len(addrs) == len(sizes):
            raise ValueError("Mooncake bulk GET requires one buffer list per key")
        if not host:
            return self._get(keys, addrs, sizes)
        results = []
        with self._recv.lock:
            for begin, end, staged in self._host_batches(addrs, sizes, self._recv):
                lengths = sizes[begin:end]
                codes = self._get(keys[begin:end], staged, lengths)
                for src, dst, spans, code in zip(staged, addrs[begin:end], lengths, codes, strict=True):
                    if code == 0:
                        for source, target, length in zip(src, dst, spans, strict=True):
                            ctypes.memmove(target, source, length)
                results.extend(codes)
        return results

    def _get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int]:
        codes = require_aligned_batch_results(
            "Mooncake bulk GET", keys, self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
        )
        # Unlike MemCache, Mooncake returns a byte count on GET. A short read
        # must fail the import; otherwise the coordinator could publish partial KV.
        return [rc if rc < 0 else (0 if rc == sum(parts) else -1) for rc, parts in zip(codes, sizes, strict=True)]
