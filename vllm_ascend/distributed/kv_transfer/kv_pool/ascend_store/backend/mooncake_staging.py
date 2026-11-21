# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded DDR staging for HiSparse Host Source pages in Mooncake Store."""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
from bisect import bisect_right
from collections.abc import Callable, Iterator
from enum import Enum, auto
from typing import Any, TypeVar

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import require_aligned_batch_results
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import BufferDescriptor, MemoryLocation

# Each direction owns a bounded buffer; network waits never serialize PUT and GET.
_HOST_STAGING_BYTES = 64 * 1024 * 1024
# Mooncake's Store API reports TRANSFER_FAIL for both a completed failure and
# an expired batch wait. Older bindings may return a generic error (-1).
_UNCONFIRMED_TRANSFER_CODES = frozenset({-800, -1})
_Result = TypeVar("_Result")


class _StoreState(Enum):
    OPEN = auto()
    DRAINING = auto()
    CLOSED = auto()


class _StoreUnavailableError(RuntimeError):
    pass


class _HostStaging:
    def __init__(self) -> None:
        self.buffer = mmap.mmap(-1, _HOST_STAGING_BYTES)
        self.base = ctypes.addressof(ctypes.c_ubyte.from_buffer(self.buffer))
        self.lock = threading.Lock()

    def prepare(self) -> None:
        # Fault pages in once at registration, outside the transfer hot path.
        # Store/HIXL owns transport registration; do not double-register with ACL.
        ctypes.memset(self.base, 0, _HOST_STAGING_BYTES)


class MooncakeHostStaging:
    """Stage Host transfers in DDR and pass device buffers directly to the Store SDK."""

    def __init__(self, store: Any, close_store: Callable[[], None]) -> None:
        self.store = store
        self._close_store = close_store
        self._store_state_changed = threading.Condition()
        self._active_store_calls = 0
        self._store_state = _StoreState.OPEN
        self._host_ranges: dict[int, int] = {}
        self._host_bases: list[int] = []
        self._registered: list[int] = []
        self._send = _HostStaging()
        try:
            self._recv = _HostStaging()
        except Exception:
            self._send.buffer.close()
            raise

    def register(self, buffers: list[BufferDescriptor]) -> None:
        if len({buffer.address for buffer in buffers}) != len(buffers):
            raise ValueError("Mooncake bulk buffer addresses must be distinct")
        self._host_ranges = {
            buffer.address: buffer.size_bytes for buffer in buffers if buffer.memory_location is MemoryLocation.HOST
        }
        if not self._host_ranges:
            raise ValueError("Mooncake bulk transfer requires explicit Host Source buffer ranges")
        self._host_bases = sorted(self._host_ranges)
        # HIXL cannot register HiSparse's already mapped/pinned pages (Resource_Busy).
        # Use ordinary DDR for Store and keep the Indexer slab on the direct device path.
        for staging in (self._send, self._recv):
            staging.prepare()
            self._register(staging.base, _HOST_STAGING_BYTES)
        for buffer in buffers:
            if buffer.memory_location is MemoryLocation.DEVICE:
                self._register(buffer.address, buffer.size_bytes)

    def _register(self, ptr: int, size: int) -> None:
        rc = self.store.register_buffer(ptr, size)
        if rc != 0:
            raise RuntimeError(f"Mooncake bulk buffer registration failed: {size} bytes, status={rc}")
        self._registered.append(ptr)

    def unregister(self) -> None:
        if self._store_state is _StoreState.CLOSED:
            return
        failures = []
        for ptr in reversed(self._registered):
            try:
                rc = self.store.unregister_buffer(ptr)
                if rc != 0:
                    logger.error("Mooncake bulk buffer unregister failed: status=%s", rc)
                    failures.append(ptr)
            except Exception:
                logger.exception("Mooncake bulk buffer unregister raised")
                failures.append(ptr)
        self._registered = failures[::-1]
        if failures:
            # Keep the buffers alive until their owner has closed the Store.
            raise RuntimeError(f"Mooncake could not unregister {len(failures)} bulk buffers")

    def close(self) -> None:
        """Release memory after successful deregistration or backend Store shutdown."""
        self._registered.clear()
        self._recv.buffer.close()
        self._send.buffer.close()

    def _call_store(self, call: Callable[[], _Result], *, transfer: bool = False) -> _Result:
        with self._store_state_changed:
            if self._store_state is not _StoreState.OPEN:
                raise _StoreUnavailableError("Mooncake Store disabled after an unconfirmed KV transfer")
            self._active_store_calls += 1
        transfer_unconfirmed = False
        try:
            return call()
        except Exception:
            transfer_unconfirmed = transfer
            raise
        finally:
            with self._store_state_changed:
                self._active_store_calls -= 1
                if transfer_unconfirmed:
                    self._store_state = _StoreState.DRAINING
                if self._store_state is _StoreState.DRAINING:
                    if self._active_store_calls == 0:
                        # The owner closes the Store's Ascend Direct engine.
                        # Keep all staging and Indexer buffers live until that
                        # call returns, including buffers used by concurrent
                        # GET and PUT calls on this worker.
                        try:
                            self._close_store()
                        except Exception:
                            logger.exception("Mooncake Store could not drain transfers; terminating worker")
                            os._exit(1)
                        self._store_state = _StoreState.CLOSED
                        self._store_state_changed.notify_all()
                    else:
                        self._store_state_changed.wait_for(lambda: self._store_state is _StoreState.CLOSED)

    def exists(self, keys: list[str]) -> list[int]:
        try:
            return self._call_store(lambda: self.store.batch_is_exist(keys))
        except _StoreUnavailableError:
            return [0] * len(keys)

    def remove_all(self) -> None:
        self._call_store(lambda: self.store.remove_all(force=True))

    def _transfer_batch(self, operation: str, keys: list[str], submit: Callable[[], list[int]]) -> list[int]:
        def checked_submit() -> list[int]:
            codes = require_aligned_batch_results(operation, keys, submit())
            if _UNCONFIRMED_TRANSFER_CODES.intersection(codes):
                raise RuntimeError(f"{operation}: Store returned with transfer completion unconfirmed")
            return codes

        return self._call_store(checked_submit, transfer=True)

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

    def _iter_host_staging_batches(
        self, addrs: list[list[int]], sizes: list[list[int]], staging: _HostStaging
    ) -> Iterator[tuple[int, int, list[list[int]]]]:
        """Yield batch slices and their staging addresses."""
        # Validate all pointers before copying. Each key fits into an empty buffer,
        # so crossing the limit always flushes a nonempty batch.
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
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        config: Any,
        *,
        memory_location: MemoryLocation,
    ) -> list[bool]:
        if not len(keys) == len(addrs) == len(sizes):
            raise ValueError("Mooncake bulk PUT requires one buffer list per key")
        if memory_location is MemoryLocation.DEVICE:
            return self._put(keys, addrs, sizes, config)
        if memory_location is not MemoryLocation.HOST:
            raise ValueError(f"Unsupported Mooncake PUT memory location: {memory_location!r}")
        result = []
        with self._send.lock:
            for begin, end, staged in self._iter_host_staging_batches(addrs, sizes, self._send):
                for host_ptrs, staging_ptrs, lengths in zip(addrs[begin:end], staged, sizes[begin:end], strict=True):
                    for host_ptr, staging_ptr, length in zip(host_ptrs, staging_ptrs, lengths, strict=True):
                        ctypes.memmove(staging_ptr, host_ptr, length)
                result.extend(self._put(keys[begin:end], staged, sizes[begin:end], config))
        return result

    def _put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]], config: Any) -> list[bool]:
        codes = self._transfer_batch(
            "Mooncake bulk PUT", keys, lambda: self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
        )
        return [rc == 0 for rc in codes]

    def get(
        self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]], *, memory_location: MemoryLocation
    ) -> list[int]:
        if not len(keys) == len(addrs) == len(sizes):
            raise ValueError("Mooncake bulk GET requires one buffer list per key")
        if memory_location is MemoryLocation.DEVICE:
            return self._get(keys, addrs, sizes)
        if memory_location is not MemoryLocation.HOST:
            raise ValueError(f"Unsupported Mooncake GET memory location: {memory_location!r}")
        results = []
        with self._recv.lock:
            for begin, end, staged in self._iter_host_staging_batches(addrs, sizes, self._recv):
                lengths = sizes[begin:end]
                codes = self._get(keys[begin:end], staged, lengths)
                for host_ptrs, staging_ptrs, spans, code in zip(addrs[begin:end], staged, lengths, codes, strict=True):
                    if code == 0:
                        for host_ptr, staging_ptr, length in zip(host_ptrs, staging_ptrs, spans, strict=True):
                            ctypes.memmove(host_ptr, staging_ptr, length)
                results.extend(codes)
        return results

    def _get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int]:
        codes = self._transfer_batch(
            "Mooncake bulk GET", keys, lambda: self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
        )
        # Unlike MemCache, Mooncake returns a byte count on GET. A short read
        # must fail the import; otherwise the coordinator could publish partial KV.
        return [rc if rc < 0 else (0 if rc == sum(parts) else -1) for rc, parts in zip(codes, sizes, strict=True)]
