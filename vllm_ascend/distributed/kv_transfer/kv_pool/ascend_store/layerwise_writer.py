# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ordered background writes of explicit per-layer object ranges.

The caller owns source buffers until finish/cancel completes. By default keys
are exclusively owned by the attempt. Explicit shared objects use ordinary
pool ownership: committed hits are borrowed under read protection; failed
idempotent allocations are never revoked as though privately owned.
"""

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import require_aligned_batch_results
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.copy_ranges import (
    copy_gva_ranges,
    range_transfer_batches,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend


@dataclass
class LayerwiseWriteCall:
    """A writer operation executed on an existing store sender queue."""

    function: Callable
    args: tuple
    future: Future

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return
        try:
            result = self.function(*self.args)
        except BaseException as exc:
            self.future.set_exception(exc)
        else:
            self.future.set_result(result)

    def cancel_queued(self):
        self.future.cancel()


class LayerwiseSenderExecutor:
    """Submit to the ordinary sender without creating another I/O queue."""

    def __init__(self, sender):
        self.sender = sender

    def submit(self, function, *args):
        future = Future()
        with self.sender._request_submit_lock:
            if self.sender._stopping.is_set() or self.sender._fatal_error is not None or not self.sender.is_alive():
                raise RuntimeError("Cannot submit a writer operation to a stopped sender")
            self.sender.request_queue.put(LayerwiseWriteCall(function, args, future))
        return future


class LayerwiseAllocationPlan:
    """Build aligned native allocation batches for both layerwise writers.

    Select keys, sizes and ownership before constructing this plan. Repeated
    keys must describe the same object; allocate them once and expand results
    back to the caller's original slots. The caller owns partial allocation,
    ambiguous RPC failures, leases and rollback. No I/O or queue is created by
    constructing the plan, and no failure here revokes a shared prefix object.
    """

    def __init__(self, keys: Sequence[str], sizes: Sequence[int], *, max_keys: int = 0):
        if len(keys) != len(sizes):
            raise ValueError("Allocation keys and sizes must have equal lengths")
        if type(max_keys) is not int or max_keys < 0:
            raise ValueError("Allocation batch limit must be a nonnegative integer")
        objects: dict[str, int] = {}
        indices: dict[str, int] = {}
        slots = []
        for key, size in zip(keys, sizes, strict=True):
            if not isinstance(key, str) or not key:
                raise ValueError("Allocation requires nonempty object keys")
            if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
                raise ValueError("Allocation requires positive integer object sizes")
            if key in objects and objects[key] != size:
                raise ValueError(f"Conflicting allocation sizes for object {key}")
            if key not in objects:
                indices[key] = len(objects)
                objects[key] = int(size)
            slots.append(indices[key])
        self.keys = tuple(objects)
        self.sizes = tuple(objects.values())
        self.slots = tuple(slots)
        self.max_keys = max_keys

    def _run(self, operation: str, start: Callable[[list[str], list[int]], list[int]]) -> list[int]:
        results: list[int] = []
        batch_size = self.max_keys or max(1, len(self.keys))
        for index in range(0, len(self.keys), batch_size):
            keys = list(self.keys[index : index + batch_size])
            sizes = list(self.sizes[index : index + batch_size])
            results.extend(require_aligned_batch_results(operation, keys, start(keys, sizes)))
        return [results[index] for index in self.slots]

    def allocate_gvas(self, allocate, lease_ttl_ms: int) -> list[int]:
        if type(lease_ttl_ms) is not int or lease_ttl_ms < 0:
            raise ValueError("Allocation lease TTL must be a nonnegative integer")
        return self._run("pool allocate", lambda keys, sizes: allocate(keys, sizes, lease_ttl_ms))

    def start_ranges(self, start) -> list[int]:
        return self._run("pool put start", start)


@dataclass(frozen=True)
class LayerWriteRange:
    key: str
    source: int
    object_offset: int
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Write range requires an object key")
        if any(type(value) is not int or value <= 0 for value in (self.source, self.size)):
            raise ValueError("Write range requires positive source address and size")
        if type(self.object_offset) is not int or self.object_offset < 0:
            raise ValueError("Write range offset must be nonnegative")


class LayerwiseWriteCompletion:
    """Track the planned layers of each object in one write attempt or step.

    Allocation and cleanup ownership stay with the caller. A shared prefix
    object and a private PD object use the same completeness check, without
    giving the shared-object caller permission to revoke somebody else's key.
    Record only after the layer's native copies have returned successfully.
    """

    def __init__(self, layers: dict[str, set[str]]):
        if not layers or any(not name or not keys for name, keys in layers.items()):
            raise ValueError("Write completion requires named layers and their object keys")
        self.layers = {name: frozenset(keys) for name, keys in layers.items()}
        self.keys = frozenset().union(*self.layers.values())
        self.completed: dict[str, set[str]] = {}

    def record(self, layer: str, keys: set[str]) -> None:
        if self.layers.get(layer) != keys:
            raise ValueError(f"Unexpected layer completion or object coverage: {layer}")
        self.record_keys(layer, keys)

    def record_keys(self, layer: str, keys: set[str]) -> None:
        """Record successful keys when a range backend reports partial failure."""
        self.validate_keys(layer, keys)
        self.completed.setdefault(layer, set()).update(keys)

    def validate_keys(self, layer: str, keys: set[str]) -> None:
        """Reject a wrong or repeated layer before touching source buffers."""
        expected = self.layers.get(layer)
        completed = self.completed.get(layer, set())
        if expected is None or not keys <= expected or keys & completed:
            raise ValueError(f"Unexpected layer completion or object coverage: {layer}")

    def require_complete(self, keys: set[str]) -> None:
        if not keys <= self.keys:
            raise ValueError("Commit contains objects outside this write plan")
        missing = sorted(
            name for name, expected in self.layers.items() if not keys & expected <= self.completed.get(name, set())
        )
        if missing:
            raise ValueError(f"Required object layers not copied: {missing}")


class LayerwiseWriteSession:
    """Synchronous layer execution shared by prefix pooling and pooled PD.

    Call on the caller's existing serial queue. A return means native copies
    have drained, so no second queue can prematurely release a source slot.
    Allocation, abort/revoke and retention remain with the object owner: a
    shared prefix allocation must not acquire private-PD rollback semantics.
    """

    def __init__(self, completion: LayerwiseWriteCompletion | None = None):
        self.completion = completion

    def _ready(self, layer, keys, event, cancelled) -> bool:
        if keys and self.completion is not None:
            self.completion.validate_keys(layer, set(keys))
        if cancelled is not None and cancelled():
            return False
        event.synchronize()
        return cancelled is None or not cancelled()

    def write_gva(
        self,
        layer,
        keys,
        event,
        copy,
        gvas,
        buffers,
        sizes,
        *,
        max_addresses=0,
        max_packet_bytes=0,
        max_batch_bytes=0,
        cancelled=None,
    ) -> int | None:
        if not self._ready(layer, keys, event, cancelled):
            return None
        result = copy_gva_ranges(
            copy,
            gvas,
            buffers,
            sizes,
            0,
            max_addresses=max_addresses,
            max_packet_bytes=max_packet_bytes,
            max_batch_bytes=max_batch_bytes,
            cancelled=cancelled,
        )
        if cancelled is not None and cancelled():
            return None
        if result == 0 and keys and self.completion is not None:
            self.completion.record_keys(layer, set(keys))
        return result

    def write_ranges(
        self,
        layer,
        keys,
        event,
        copy,
        buffers,
        sizes,
        offsets,
        *,
        max_keys=0,
        max_packet_bytes=0,
        max_batch_bytes=0,
        max_batch_ranges=0,
        stop_on_error=True,
        cancelled=None,
    ) -> list[int] | None:
        if len(set(keys)) != len(keys):
            raise ValueError("A layer range write requires unique object keys")
        if not self._ready(layer, keys, event, cancelled):
            return None
        results = dict.fromkeys(keys, 0)
        for batch_keys, batch_buffers, batch_sizes, batch_offsets in range_transfer_batches(
            keys,
            buffers,
            sizes,
            offsets,
            max_keys,
            max_packet_bytes,
            max_batch_bytes=max_batch_bytes,
            max_batch_ranges=max_batch_ranges,
        ):
            if cancelled is not None and cancelled():
                return None
            codes = require_aligned_batch_results(
                "Pool layer copy", batch_keys, copy(batch_keys, batch_buffers, batch_sizes, batch_offsets)
            )
            if stop_on_error and any(code < 0 for code in codes):
                raise RuntimeError(f"Pool layer copy failed: {codes}")
            if cancelled is not None and cancelled():
                return None
            for key, code in zip(batch_keys, codes, strict=True):
                if results[key] >= 0:
                    results[key] = code if code < 0 else results[key] + code
        if cancelled is not None and cancelled():
            return None
        if keys and self.completion is not None:
            self.completion.record_keys(layer, {key for key, code in results.items() if code >= 0})
        return [results[key] for key in keys]

    def publish(self, keys: list[str], finish: Callable[[list[str]], list[int]]) -> list[int]:
        if self.completion is not None:
            self.completion.require_complete(set(keys))
        return require_aligned_batch_results("pool write commit", keys, finish(keys))


class LayerwiseWriteHandle:
    def __init__(
        self, objects: dict[str, int], layers: dict[str, tuple[LayerWriteRange, ...]], *, padded_objects=frozenset()
    ):
        self.objects = dict(objects)
        self.layers = {name: tuple(ranges) for name, ranges in layers.items()}
        if not self.objects or not self.layers:
            raise ValueError("Layerwise writes require objects and expected layers")
        coverage: dict[str, list[tuple[int, int]]] = {key: [] for key in self.objects}
        for name, ranges in self.layers.items():
            if not isinstance(name, str) or not name or not ranges:
                raise ValueError("Expected write layers must be named and nonempty")
            for item in ranges:
                if item.key not in coverage:
                    raise ValueError(f"Unknown write object: {item.key}")
                coverage[item.key].append((item.object_offset, item.object_offset + item.size))
        for key, size in self.objects.items():
            if not isinstance(key, str) or not key or type(size) is not int or size <= 0:
                raise ValueError("Write objects require keys and positive byte sizes")
            end = 0
            for start, stop in sorted(coverage[key]):
                if start < end or (start != end and key not in padded_objects):
                    raise ValueError(f"Incomplete or overlapping write coverage for {key}")
                end = stop
            if end > size or (end != size and key not in padded_objects):
                raise ValueError(f"Write coverage does not match object size for {key}")
        self.completion = LayerwiseWriteCompletion(
            {name: {item.key for item in ranges} for name, ranges in self.layers.items()}
        )
        self.session = LayerwiseWriteSession(self.completion)
        self.submitted: set[str] = set()
        self.cancelled = threading.Event()
        self.gvas: dict[str, int] = {}
        self.started: list[str] = []
        self.untracked_allocations: list[str] = []
        self.copy_futures: list[Future[None]] = []
        self.terminal: Future[None] | None = None
        self.failure: BaseException | None = None
        self.lease_deadline: float | None = None
        self.shared_objects: frozenset[str] = frozenset()
        self.borrowed: set[str] = set()
        self.borrowed_leases: list[str] = []
        self.finalized = False


class LayerwiseWriter:
    def __init__(
        self,
        backend: "Backend",
        *,
        lease_ttl_ms: int,
        max_bytes: int = 64 << 20,
        clock: Callable[[], float] = time.monotonic,
        executor=None,
    ):
        if backend.layerwise_data_plane not in ("gva", "block_key"):
            raise ValueError("Backend cannot write layerwise object ranges")
        if type(lease_ttl_ms) is not int or lease_ttl_ms <= 0 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Writer lease TTL and copy limit must be positive integers")
        self.backend = backend
        self.lease_ttl_ms = lease_ttl_ms
        self.max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.RLock()
        self._active: dict[str, LayerwiseWriteHandle] = {}
        self._handles: set[LayerwiseWriteHandle] = set()
        self._closed = False
        self._cleanup_error: BaseException | None = None
        self._owns_queue = executor is None
        self._queue = (
            executor
            if executor is not None
            else ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pool-layer-write", initializer=backend.set_device
            )
        )

    def prepare(
        self,
        objects: dict[str, int],
        layers: dict[str, tuple[LayerWriteRange, ...]],
        *,
        shared_objects: frozenset[str] = frozenset(),
    ) -> LayerwiseWriteHandle:
        handle = LayerwiseWriteHandle(objects, layers, padded_objects=shared_objects)
        if not shared_objects <= objects.keys():
            raise ValueError("Shared objects must belong to the write plan")
        if shared_objects and self.backend.layerwise_data_plane != "gva":
            raise ValueError("Shared range sessions require backend write-ownership support")
        handle.shared_objects = frozenset(shared_objects)
        private = objects.keys() - shared_objects
        with self._lock:
            if self._closed:
                raise RuntimeError("Layerwise writer is closed")
            if self._cleanup_error is not None:
                raise RuntimeError("Layerwise writer has an unresolved cleanup failure") from self._cleanup_error
            if self._active.keys() & private:
                raise ValueError("A pool object already has an active writer")
            self._active.update((key, handle) for key in private)
            self._handles.add(handle)
            try:
                handle.copy_futures.append(self._queue.submit(self._begin, handle))
            except BaseException as exc:
                handle.failure = exc
                handle.cancelled.set()
                raise
        return handle

    def _begin(self, handle: LayerwiseWriteHandle) -> None:
        try:
            keys = list(handle.objects)
            sizes = list(handle.objects.values())
            if handle.shared_objects:
                shared = [key for key in keys if key in handle.shared_objects]
                readable = self.backend.batch_is_readable(shared)
                if len(readable) != len(shared) or any(type(value) is not bool for value in readable):
                    raise ValueError("Shared pool readability response is not aligned")
                borrowed = [key for key, hit in zip(shared, readable, strict=True) if hit]
                if borrowed:
                    # Record before the RPC: an ambiguous response can still
                    # acquire native leases and must retain cleanup ownership.
                    handle.borrowed_leases = borrowed
                    codes = require_aligned_batch_results(
                        "shared pool read protection",
                        borrowed,
                        self.backend.batch_add_lease(borrowed, self.lease_ttl_ms),
                    )
                    if any(code != 0 for code in codes):
                        raise RuntimeError(f"Shared pool acquisition failed: {codes}")
                    infos = self.backend.batch_get_key_info(borrowed, for_load=True)
                    if infos is None or len(infos) != len(borrowed):
                        raise ValueError("Shared pool object metadata is not aligned")
                    if any(
                        info is None or int(info.size()) != handle.objects[key]
                        for key, info in zip(borrowed, infos, strict=True)
                    ):
                        raise ValueError("Shared pool object size does not match the source layout")
                    handle.borrowed = set(borrowed)
                    keys = [key for key in keys if key not in handle.borrowed]
                    sizes = [handle.objects[key] for key in keys]
            allocation = LayerwiseAllocationPlan(keys, sizes)
            # A malformed response may still have created native sessions.
            # Keys are exclusively owned by this attempt, so rollback must also
            # cover calls which raised before returning per-key status.
            if self.backend.layerwise_data_plane == "gva":
                # Start before the RPC: allocation latency consumes protection
                # too. Never assume an uncommitted object can use AddLease.
                handle.lease_deadline = self._clock() + self.lease_ttl_ms / 1000
                # A failed local GVA registration can follow a successful
                # remote allocation. Keep even invalid/unknown results owned
                # until private-key revocation confirms cleanup.
                handle.untracked_allocations = keys
                results = allocation.allocate_gvas(self.backend.batch_alloc, self.lease_ttl_ms)
                handle.started = [key for key, gva in zip(keys, results, strict=True) if 0 < gva < (1 << 64) - 1]
                handle.untracked_allocations = [key for key in keys if key not in handle.started]
                if len(handle.started) != len(keys):
                    raise RuntimeError("Pool object allocation failed or returned an SSD object")
                handle.gvas = dict(zip(keys, results, strict=True))
                self._check_write_lease(handle)
            else:
                handle.started = keys
                results = allocation.start_ranges(self.backend.batch_put_start)
                handle.started = [key for key, code in zip(keys, results, strict=True) if code == 0]
                if len(handle.started) != len(keys):
                    raise RuntimeError(f"Pool put start failed: {results}")
        except BaseException as exc:
            handle.failure = exc
            raise

    def _check_write_lease(self, handle: LayerwiseWriteHandle) -> None:
        if handle.lease_deadline is not None and self._clock() >= handle.lease_deadline:
            raise TimeoutError("Pool write lease expired before the attempt could commit")

    def submit_layer(self, handle: LayerwiseWriteHandle, layer_name: str, ready_event: Any) -> None:
        with self._lock:
            if self._closed or handle.terminal is not None or handle.cancelled.is_set():
                raise RuntimeError("Write attempt is closing or cancelled")
            if handle not in self._handles:
                raise ValueError("Write handle does not belong to this writer")
            if layer_name not in handle.layers or layer_name in handle.submitted:
                raise ValueError(f"Unexpected or duplicate layer write: {layer_name}")
            if not callable(getattr(ready_event, "synchronize", None)):
                raise ValueError("Layer write requires a recorded compute completion event")
            handle.submitted.add(layer_name)
            handle.copy_futures.append(self._queue.submit(self._copy, handle, layer_name, ready_event))

    def _copy(self, handle: LayerwiseWriteHandle, layer_name: str, event: Any) -> None:
        try:
            if handle.failure is not None or handle.cancelled.is_set():
                return
            if self.backend.layerwise_data_plane == "gva":
                planned = handle.layers[layer_name]
                ranges = tuple(item for item in planned if item.key not in handle.borrowed)

                def copy(gvas, buffers, sizes, direction):
                    self._check_write_lease(handle)
                    result = self.backend.batch_copy_gva(gvas, buffers, sizes, direction)
                    if result != 0:
                        return result
                    # The SDK may only validate a write lease before I/O.
                    # Drain that I/O before failing; do not publish stale data
                    # or submit subsequent batches after protection expires.
                    self._check_write_lease(handle)
                    return result

                code = handle.session.write_gva(
                    layer_name,
                    {item.key for item in planned},
                    event,
                    copy,
                    [handle.gvas[item.key] + item.object_offset for item in ranges],
                    [item.source for item in ranges],
                    [item.size for item in ranges],
                    max_addresses=256,
                    max_packet_bytes=self.max_bytes,
                    max_batch_bytes=self.max_bytes,
                    cancelled=handle.cancelled.is_set,
                )
                if code is None:
                    return
                if code != 0:
                    raise RuntimeError(f"Pool layer copy failed: {code}")
                return
            grouped: dict[str, list[LayerWriteRange]] = {}
            for item in handle.layers[layer_name]:
                grouped.setdefault(item.key, []).append(item)
            keys = list(grouped)
            handle.session.write_ranges(
                layer_name,
                keys,
                event,
                self.backend.batch_copy_put,
                [[item.source for item in grouped[key]] for key in keys],
                [[item.size for item in grouped[key]] for key in keys],
                [[item.object_offset for item in grouped[key]] for key in keys],
                max_keys=256,
                max_packet_bytes=self.max_bytes,
                max_batch_ranges=256,
                max_batch_bytes=self.max_bytes,
                cancelled=handle.cancelled.is_set,
            )
        except BaseException as exc:
            handle.failure = exc
            raise

    def finish(self, handle: LayerwiseWriteHandle) -> Future[None]:
        with self._lock:
            if handle.terminal is not None:
                return handle.terminal
            if handle.submitted != handle.layers.keys():
                raise ValueError(f"Required layers not submitted: {handle.layers.keys() - handle.submitted}")
            handle.terminal = self._queue.submit(self._finish, handle)
            return handle.terminal

    def cancel(self, handle: LayerwiseWriteHandle) -> Future[None]:
        with self._lock:
            handle.cancelled.set()
            if handle.terminal is None:
                handle.terminal = self._queue.submit(self._finish, handle)
            return handle.terminal

    def _abort(self, handle: LayerwiseWriteHandle) -> None:
        self._release_borrowed(handle)
        # Shared allocation is idempotent, not an exclusive-create claim.
        # Match ordinary prefix pooling: an unsuccessful writer never removes
        # a key another producer may have finished. Unpublished native writes
        # remain subject to their allocation lease and pool eviction.
        handle.untracked_allocations = [key for key in handle.untracked_allocations if key not in handle.shared_objects]
        handle.started = [key for key in handle.started if key not in handle.shared_objects]
        if handle.untracked_allocations:
            keys = handle.untracked_allocations
            codes = require_aligned_batch_results("pool allocation revoke", keys, self.backend.batch_revoke(keys))
            handle.untracked_allocations = [key for key, code in zip(keys, codes, strict=True) if code != 0]
            if handle.untracked_allocations:
                raise RuntimeError(f"Pool allocation cleanup failed: {codes}")
        if not handle.started:
            return
        if self.backend.layerwise_data_plane == "gva":
            raw = self.backend.batch_write_finish(handle.started, [-1] * len(handle.started))
        else:
            raw = self.backend.batch_revoke(handle.started)
        codes = require_aligned_batch_results("pool write abort", handle.started, raw)
        # Successful cleanup can remove the SDK's local allocation/session
        # record. Retrying those keys is not necessarily idempotent. Record
        # progress before the expired-lease fallback, which may itself fail or
        # return an ambiguous response.
        handle.started = [key for key, code in zip(handle.started, codes, strict=True) if code != 0]
        if (
            self.backend.layerwise_data_plane == "gva"
            and handle.lease_deadline is not None
            and self._clock() >= handle.lease_deadline
            and handle.started
        ):
            # Expiry can remove the SDK's local allocation tracker, making
            # WRITE_FAIL unusable. I/O has drained and these unpublished
            # keys belong exclusively to this attempt, so delete them by
            # key. Never apply this fallback to shared prefix allocations.
            keys = handle.started
            codes = require_aligned_batch_results("pool expired write revoke", keys, self.backend.batch_revoke(keys))
            handle.started = [key for key, code in zip(keys, codes, strict=True) if code != 0]
        if handle.started:
            raise RuntimeError(f"Pool write cleanup failed: {codes}")

    def _release_borrowed(self, handle: LayerwiseWriteHandle) -> None:
        if handle.borrowed_leases:
            code = self.backend.batch_remove_lease(handle.borrowed_leases)
            if type(code) is not int or code != 0:
                raise RuntimeError(f"Shared pool lease release failed: {code}")
            handle.borrowed_leases = []

    def _finish(self, handle: LayerwiseWriteHandle) -> None:
        finalized = False
        try:
            if handle.failure is not None:
                raise RuntimeError("Pool write attempt failed") from handle.failure
            if handle.cancelled.is_set():
                raise RuntimeError("Pool write attempt cancelled")
            self._check_write_lease(handle)
            keys = [key for key in handle.objects if key not in handle.borrowed]
            finish = (
                (lambda keys: self.backend.batch_write_finish(keys, [0] * len(keys)))
                if self.backend.layerwise_data_plane == "gva"
                else self.backend.batch_commit
            )
            handle.completion.require_complete(set(handle.objects))
            codes = handle.session.publish(keys, finish) if keys else []
            if any(code != 0 for code in codes):
                raise RuntimeError(f"Pool write commit failed: {codes}")
            self._release_borrowed(handle)
            finalized = True
        except BaseException:
            try:
                self._abort(handle)
                finalized = True
            except BaseException as exc:
                self._cleanup_error = exc
                raise
            raise
        finally:
            if finalized:
                self._forget(handle)

    def _forget(self, handle: LayerwiseWriteHandle) -> None:
        with self._lock:
            handle.finalized = True
            self._handles.discard(handle)
            for key in handle.objects:
                if self._active.get(key) is handle:
                    del self._active[key]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            handles = set(self._handles)
            for handle in handles:
                try:
                    self.cancel(handle)
                except RuntimeError as exc:
                    # An externally owned sender can already have stopped.
                    # Its queued operations resolve as cancelled; drain all
                    # recorded work before retrying cleanup on this thread.
                    handle.failure = exc
        if self._owns_queue:
            self._queue.shutdown(wait=True)
        else:
            for handle in handles:
                futures = handle.copy_futures + ([handle.terminal] if handle.terminal is not None else [])
                for future in futures:
                    # Failure/cleanup ownership is retained on the handle;
                    # draining must continue through every queued operation.
                    with suppress(BaseException):
                        future.result()
        # All native copies have drained. Failed rollback still owns its keys,
        # even though its terminal future has already reported an error.
        # Retry on the closing thread with the backend's device context set.
        with self._lock:
            pending = set(self._handles)
        if pending:
            self.backend.set_device()
            for handle in pending:
                try:
                    self._abort(handle)
                except BaseException as exc:
                    self._cleanup_error = exc
                else:
                    self._forget(handle)
        with self._lock:
            if not self._handles:
                self._cleanup_error = None
        if self._cleanup_error is not None:
            raise RuntimeError("Layerwise writer cleanup failed") from self._cleanup_error
