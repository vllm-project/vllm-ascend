# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MemCache object sessions shared by bulk and layerwise Store execution."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import PoolObject


class MemCacheReadLeaseError(RuntimeError):
    """The SDK returned failures for individual read leases."""

    def __init__(self, statuses: set[int]):
        self.statuses = frozenset(statuses)
        super().__init__(f"MemCache read lease failed with statuses {sorted(statuses)}")


@dataclass
class StoreIOSession:
    plans: tuple[PoolObject, ...]
    gvas: tuple[int, ...]
    deadline: float
    writing: bool
    completed: set[tuple[int, int]] = field(default_factory=set)
    released: bool = False
    closed: bool = False


class MemcacheStoreLayoutIO:
    """Execute upstream object ranges without changing their keys or byte order.

    Calls are synchronous. A session retains a lease until all native copies
    return. Deadlines bound submission; native in-flight lease protection still
    depends on the installed SDK contract (not implied by these checks).
    """

    def __init__(self, backend, lease_ttl_ms: int = 600000, max_segments: int = 1024):
        if lease_ttl_ms <= 0 or max_segments <= 0:
            raise ValueError("Positive lease duration and copy batch size are required")
        self.backend = backend
        self.lease_ttl_ms = lease_ttl_ms
        self.max_segments = max_segments
        # A batch can restore the same immutable object into several requests.
        # MemCache leases belong to the client/key, not to each local session.
        self._references: dict[str, int] = {}
        self._deadlines: dict[str, float] = {}
        self._pending_release: set[str] = set()

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        # Allocation alone is not a hit: only published objects are readable.
        values = self.backend.batch_is_readable(keys)
        if len(values) != len(keys):
            raise RuntimeError("MemCache returned an incomplete readability response")
        return [int(value) for value in values]

    def begin(self, plans: Sequence[PoolObject], *, writing: bool) -> StoreIOSession:
        plans = tuple(plans)
        keys = [plan.key for plan in plans]
        if len(set(keys)) != len(keys):
            raise ValueError("One session must contain unique object keys")
        deadline = time.monotonic() + self.lease_ttl_ms / 1000
        if writing:
            exists = self.batch_is_exist(keys)
            plans = tuple(plan for plan, hit in zip(plans, exists, strict=True) if not hit)
            keys = [plan.key for plan in plans]
        if any(key in self._pending_release for key in keys):
            raise RuntimeError("Previous MemCache lease cleanup must succeed before reuse")
        if writing and any(key in self._references for key in keys):
            raise RuntimeError("An unpublished object already has an active local writer")
        fresh = [key for key in keys if key not in self._references]
        for key in fresh:
            self._deadlines[key] = deadline
        deadline = min((self._deadlines[key] for key in keys), default=deadline)
        for key in keys:
            self._references[key] = self._references.get(key, 0) + 1
        session = StoreIOSession(plans, (), deadline, writing)
        try:
            if writing:
                gvas = self.backend.batch_alloc(keys, [plan.size for plan in plans], self.lease_ttl_ms) if keys else []
            else:
                # AddLease returns status codes, not addresses. Shared reads
                # keep the first lease until their last local consumer finishes.
                results = self.backend.batch_add_lease(fresh, self.lease_ttl_ms) if fresh else []
                gvas = []
                if len(results) == len(fresh):
                    # Failed AddLease entries acquired no lease. Removing
                    # them is an SDK error, not cleanup, and would poison the
                    # client's pending-release set after an ordinary miss.
                    failed = {key for key, result in zip(fresh, results, strict=True) if result != 0}
                    if failed:
                        session.plans = tuple(plan for plan in plans if plan.key not in failed)
                        for key in failed:
                            del self._references[key]
                            del self._deadlines[key]
                if len(results) != len(fresh):
                    raise RuntimeError("MemCache returned an incomplete read lease response")
                if statuses := {result for result in results if result != 0}:
                    raise MemCacheReadLeaseError(statuses)
            if not writing and keys:
                infos = self.backend.batch_get_key_info(keys, for_load=True)
                if len(infos) != len(plans) or any(
                    info is None or info.size() != plan.size for info, plan in zip(infos, plans, strict=True)
                ):
                    raise RuntimeError("MemCache object size differs from the local StoreLayout")
                gvas = [info.gva_list()[0] for info in infos]
            if len(gvas) != len(plans) or any(not 0 < int(gva) < (1 << 64) - 1 for gva in gvas):
                raise RuntimeError("MemCache allocation/read lease failed")
            session.gvas = tuple(int(gva) for gva in gvas)
            return session
        except Exception as error:
            try:
                self.abort(session)
            except Exception as cleanup:
                error.add_note(f"Lease cleanup also failed: {cleanup}")
            raise

    def copy(self, session: StoreIOSession, layers: set[str] | None = None) -> None:
        if session.closed or session.released or time.monotonic() >= session.deadline:
            raise RuntimeError("Store session is closed or its lease deadline elapsed")
        selected = [
            (object_index, range_index, part)
            for object_index, plan in enumerate(session.plans)
            for range_index, part in enumerate(plan.ranges)
            if (layers is None or part.owner in layers) and (object_index, range_index) not in session.completed
        ]
        for start in range(0, len(selected), self.max_segments):
            batch = selected[start : start + self.max_segments]
            if time.monotonic() >= session.deadline:
                raise RuntimeError("Store lease deadline elapsed before native copy")
            result = self.backend.store.batch_copy(
                [session.gvas[i] + part.offset for i, _, part in batch],
                [part.address for _, _, part in batch],
                [part.size for _, _, part in batch],
                0 if session.writing else 1,
            )
            if result != 0 or time.monotonic() >= session.deadline:
                raise RuntimeError(f"MemCache copy failed or expired: result={result}")
            session.completed.update((i, j) for i, j, _ in batch)

    def finish(self, session: StoreIOSession) -> None:
        if session.closed:
            return
        try:
            expected = sum(len(plan.ranges) for plan in session.plans)
            if len(session.completed) != expected:
                raise RuntimeError("Cannot finish a partially copied Store object")
            if time.monotonic() >= session.deadline:
                raise RuntimeError("Store lease expired before publication")
            keys = [plan.key for plan in session.plans]
            if session.writing and keys:
                results = self.backend.batch_write_finish(keys, [0] * len(keys))
                if len(results) != len(keys) or any(result != 0 for result in results):
                    raise RuntimeError("MemCache publication failed")
        except Exception as error:
            try:
                self.abort(session)
            except Exception as cleanup:
                error.add_note(f"Lease cleanup also failed: {cleanup}")
            raise
        self.abort(session)

    def abort(self, session: StoreIOSession) -> None:
        if session.closed:
            return
        if not session.released:
            for plan in session.plans:
                self._references[plan.key] -= 1
                if self._references[plan.key] == 0:
                    del self._references[plan.key]
                    # WRITE_OK already consumes the native write lease. Failed
                    # writes retain their allocation lease until backend TTL;
                    # remove_lease is a read-lease API, not a write rollback.
                    if not session.writing:
                        self._pending_release.add(plan.key)
                    else:
                        del self._deadlines[plan.key]
            session.released = True
        self.retry_cleanup()
        session.closed = True

    def retry_cleanup(self) -> None:
        if self._pending_release:
            # Do not WRITE_FAIL/delete: allocation may race with another writer.
            # Transfers have drained before release. Match the ordinary pool:
            # a returned cleanup failure is diagnostic, not a transfer failure.
            # Let backend TTL reclaim leftover leases. Do not retry old keys
            # after a new session may have acquired a lease for the same key.
            keys = sorted(self._pending_release)
            result = self.backend.batch_remove_lease(keys)
            if result != 0:
                logger.error(
                    "Shared pool lease cleanup returned %s for %d keys; backend TTL will reclaim leases",
                    result,
                    len(keys),
                )
            self._pending_release.difference_update(keys)
            for key in keys:
                del self._deadlines[key]
