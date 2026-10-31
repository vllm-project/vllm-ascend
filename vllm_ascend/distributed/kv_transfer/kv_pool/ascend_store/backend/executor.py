# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synchronous object-range restore shared by bulk and PD readers.

One executor owns read sessions for one backend client. Overlapping restores
share a native per-key session until their last synchronous copy completes.
The caller must supply committed objects and registered destination buffers.
"""

import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import require_aligned_batch_results
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.copy_ranges import (
    copy_gva_ranges,
    copy_key_ranges,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend


@dataclass(frozen=True)
class PoolReadRange:
    key: str
    object_size: int
    source_offset: int
    destination: int
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Pool range requires a source key")
        for name in ("object_size", "destination", "size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Pool range {name} must be a positive integer")
        if type(self.source_offset) is not int or not 0 <= self.source_offset <= self.object_size - self.size:
            raise ValueError("Pool read range exceeds its source object")


class PoolBackendExecutor:
    def __init__(self, backend: "Backend", *, lease_ttl_ms: int, max_ranges: int = 256, max_bytes: int = 64 << 20):
        if type(lease_ttl_ms) is not int or lease_ttl_ms <= 0:
            raise ValueError("Pool reader requires a positive lease TTL")
        if max_ranges <= 0 or max_bytes <= 0:
            raise ValueError("Pool copy batch limits must be positive")
        if backend.layerwise_data_plane not in ("gva", "block_key"):
            raise ValueError("Backend does not support layerwise range reads")
        self.backend = backend
        self.lease_ttl_ms = lease_ttl_ms
        self.max_ranges = max_ranges
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._readers: Counter[str] = Counter()
        self._lease_counts: Counter[str] = Counter()
        self._pending_release: set[str] = set()
        self._release_error: Exception | None = None
        self._lease_error: Exception | None = None
        self._closed = False
        self._renew_wakeup = threading.Event()
        self._renew_thread: threading.Thread | None = None

    def _renew_leases(self) -> None:
        # AddLease creates a native operation ID. Acquire a new lease BEFORE
        # retiring the oldest one; otherwise periodic renewal leaks native
        # leases or creates an unprotected interval. Final release uses the
        # same lock and drains every outstanding acquisition after copies end.
        try:
            self.backend.set_device()
            while True:
                self._renew_wakeup.wait(self.lease_ttl_ms / 3000)
                self._renew_wakeup.clear()
                with self._lock:
                    if self._closed or not self._readers:
                        self._renew_thread = None
                        return
                    keys = list(self._readers)
                    self._lease_counts.update(keys)
                    codes = require_aligned_batch_results(
                        "pool read lease renewal", keys, self.backend.batch_add_lease(keys, self.lease_ttl_ms)
                    )
                    self._discard_unacquired_leases(keys, codes)
                    if any(code != 0 for code in codes):
                        raise RuntimeError(f"Pool read lease renewal failed: {codes}")
                    self._release_one(keys)
        except Exception as exc:
            with self._lock:
                self._lease_error = exc
        finally:
            with self._lock:
                if self._renew_thread is threading.current_thread():
                    self._renew_thread = None

    def close(self) -> None:
        """Call after readers drain, before closing their backend client."""
        with self._lock:
            if self._readers:
                # An early close must not stop renewal for in-flight copies.
                raise RuntimeError("Pool read sessions must drain before closing their executor")
            self._closed = True
            thread = self._renew_thread
            self._renew_wakeup.set()
        if thread is not None:
            thread.join()
        with self._lock:
            if self._pending_release:
                self.backend.set_device()
                self._release_or_track(sorted(self._pending_release))
            self._release_error = None

    def _release_one(self, keys: list[str]) -> None:
        if not keys:
            return
        result = (
            self.backend.batch_remove_lease(keys)
            if self.backend.layerwise_data_plane == "gva"
            else self.backend.batch_get_end(keys)
        )
        if result != 0:
            raise RuntimeError(f"Pool read protection release failed: result={result}, keys={keys}")
        self._lease_counts.subtract(keys)
        for key in keys:
            if not self._lease_counts[key]:
                del self._lease_counts[key]

    def _discard_unacquired_leases(self, keys: list[str], results: list[int]) -> None:
        # Call only after validating the complete response. A missing key is
        # not an uncertain acquisition; attempting to release it can poison
        # an otherwise healthy reader. Preserve earlier leases on renewal.
        if not any(results):
            return
        absent_keys = self.backend.absent_read_lease_keys(keys, results)
        for key in keys:
            if key in absent_keys:
                self._lease_counts[key] -= 1
                if not self._lease_counts[key]:
                    del self._lease_counts[key]

    def _release_native(self, keys: list[str]) -> None:
        while pending := [key for key in keys if self._lease_counts[key] > 0]:
            self._release_one(pending)

    def _release_or_track(self, keys: list[str]) -> None:
        """Keep uncertain native ownership separate from active copy counts."""
        try:
            self._release_native(keys)
        except Exception as exc:
            self._pending_release.update(keys)
            self._release_error = exc
            raise
        else:
            self._pending_release.difference_update(keys)

    def _finish_read(self, keys: list[str]) -> None:
        ending = [key for key in keys if self._readers[key] == 1]
        self._readers.subtract(keys)
        for key in ending:
            del self._readers[key]
        if not self._readers:
            self._renew_wakeup.set()
        self._release_or_track(ending)

    def acquire_read_protection(self, keys: list[str]) -> None:
        """Retain committed objects across copies or successive prefill chunks."""
        # Serializing start/end prevents one request ending a native process-key
        # session while another request is entering it. Copies run outside lock.
        with self._lock:
            if self._closed:
                raise RuntimeError("Pool executor is closed")
            if self._release_error is not None:
                raise RuntimeError("Pool executor has an unresolved read protection release") from self._release_error
            if self._lease_error is not None:
                raise RuntimeError("Pool executor has an unresolved lease renewal failure") from self._lease_error
            new_keys = [key for key in keys if not self._readers[key]]
            if new_keys:
                # An exception or malformed response can follow a partial
                # native acquisition. Retain one cleanup obligation per key.
                self._lease_counts.update(new_keys)
                try:
                    raw = (
                        self.backend.batch_add_lease(new_keys, self.lease_ttl_ms)
                        if self.backend.layerwise_data_plane == "gva"
                        else self.backend.batch_get_start(new_keys)
                    )
                    results = require_aligned_batch_results("pool read start", new_keys, raw)
                    self._discard_unacquired_leases(new_keys, results)
                    if any(result != 0 for result in results):
                        raise RuntimeError(f"Pool read protection acquisition failed: {results}")
                except Exception:
                    self._release_or_track(new_keys)
                    raise
            self._readers.update(keys)
            if self.backend.layerwise_data_plane == "gva" and self._renew_thread is None:
                self._renew_thread = threading.Thread(target=self._renew_leases, name="pool-read-leases", daemon=True)
                try:
                    self._renew_thread.start()
                except Exception:
                    self._renew_thread = None
                    self._finish_read(keys)
                    raise

    def release_read_protection(self, keys: list[str]) -> None:
        """Consume these logical references even when native release fails.

        Uncertain native releases stay executor-owned for a later retry.
        """
        with self._lock:
            if any(self._readers[key] <= 0 for key in keys):
                raise ValueError("Pool read protection was already released")
            self._finish_read(keys)
            self.check_read_protection()

    def check_read_protection(self) -> None:
        with self._lock:
            if self._lease_error is not None:
                raise RuntimeError("Pool read lease renewal failed during restore") from self._lease_error

    def retry_pending_releases(self) -> None:
        with self._lock:
            if self._pending_release:
                self.backend.set_device()
                self._release_or_track(sorted(self._pending_release))
            self._release_error = None

    @contextmanager
    def _reading(self, keys: list[str]) -> "Iterator[None]":
        self.acquire_read_protection(keys)
        try:
            yield
        finally:
            self.release_read_protection(keys)

    def _copy_gva(self, gvas: list[int], buffers: list[int], sizes: list[int], direction: int) -> int:
        # Renewal can fail while metadata lookup or a previous native batch is
        # in flight. Drain that call, then stop submitting more copies. Do not
        # hold this lock over I/O: the renewal thread needs it to protect an
        # active batch. The SDK still owns protection during that batch.
        with self._lock:
            if self._lease_error is not None:
                raise RuntimeError("Pool read lease renewal failed before copy submission") from self._lease_error
        return self.backend.batch_copy_gva(gvas, buffers, sizes, direction)

    def restore(self, ranges: list[PoolReadRange]) -> None:
        """Restore every range or raise; no successful partial-request result."""
        if not ranges:
            return
        object_sizes: dict[str, int] = {}
        for item in ranges:
            previous = object_sizes.setdefault(item.key, item.object_size)
            if previous != item.object_size:
                raise ValueError(f"Conflicting source object sizes for {item.key}")
        keys = list(object_sizes)
        with self._reading(keys):
            bases: dict[str, int] = {}
            if self.backend.layerwise_data_plane == "gva":
                infos = self.backend.batch_get_key_info(keys, for_load=True)
                if infos is None or len(infos) != len(keys):
                    raise RuntimeError("Pool key metadata response length mismatch")
                for key, info in zip(keys, infos, strict=True):
                    if info is None or info.size() != object_sizes[key]:
                        raise RuntimeError(f"Pool source object size mismatch: {key}")
                    addresses = info.gva_list()
                    if not addresses or int(addresses[0]) <= 0:
                        raise RuntimeError(f"Pool source object has no readable GVA: {key}")
                    bases[key] = int(addresses[0])
                result = copy_gva_ranges(
                    self._copy_gva,
                    [bases[item.key] + item.source_offset for item in ranges],
                    [item.destination for item in ranges],
                    [item.size for item in ranges],
                    1,
                    max_addresses=self.max_ranges,
                    max_packet_bytes=self.max_bytes,
                    max_batch_bytes=self.max_bytes,
                )
                if result != 0:
                    raise RuntimeError(f"Pool range copy failed: result={result}")
                return
            grouped: dict[str, list[PoolReadRange]] = {}
            for item in ranges:
                grouped.setdefault(item.key, []).append(item)
            copy_key_ranges(
                self.backend.batch_copy_get,
                keys,
                [[item.destination for item in grouped[key]] for key in keys],
                [[item.size for item in grouped[key]] for key in keys],
                [[item.source_offset for item in grouped[key]] for key in keys],
                max_ranges=self.max_ranges,
                max_bytes=self.max_bytes,
                operation="Pool range copy",
            )
