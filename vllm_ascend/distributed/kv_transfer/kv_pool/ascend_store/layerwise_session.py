# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import RequestChunkKey


class LayerwiseStoreBackend(Protocol):
    def exists(self, keys: list[str]) -> list[int]: ...

    def batch_put_session_start(
        self, keys: list[str], sizes: list[int], config=None
    ) -> list[int]: ...

    def batch_put_from_multi_buffer_ranges(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_dst_offsets: list[list[int]],
    ) -> list[int]: ...

    def batch_put_session_end(self, keys: list[str]) -> list[int]: ...

    def batch_put_session_revoke(self, keys: list[str]) -> list[int]: ...

    def batch_get_session_start(self, keys: list[str]) -> list[int]: ...

    def batch_get_into_multi_buffer_ranges(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_src_offsets: list[list[int]],
    ) -> list[int]: ...

    def batch_get_session_end(self, keys: list[str]) -> int: ...


# Mooncake native error codes consumed by the put claim path. The value must
# stay aligned with ``ErrorCode`` in mooncake-store/include/types.h.
MOONCAKE_OBJECT_ALREADY_EXISTS = -705


class PutState(str, Enum):
    OPEN = "open"
    COMMITTED = "committed"
    REVOKED = "revoked"
    END_FAILED = "end_failed"
    CLEANUP_FAILED = "cleanup_failed"
    ERROR = "error"


class StoreCommitError(RuntimeError):
    pass


@dataclass(frozen=True)
class PutBinding:
    key: str
    object_size: int

    def __post_init__(self) -> None:
        if not self.key or self.object_size <= 0:
            raise ValueError("PutBinding requires a key and positive object size")


@dataclass
class PutClaim:
    owned_keys: list[str] = field(default_factory=list)
    followed_keys: list[str] = field(default_factory=list)
    failed_keys: list[str] = field(default_factory=list)
    # Keys whose put_start hit OBJECT_ALREADY_EXISTS from a cross-instance
    # owner. The local execution yielded write ownership: no entry, no
    # future, no wait. Observability only; not part of any control flow.
    yielded_keys: list[str] = field(default_factory=list)
    commit_futures: dict[str, Future[list[int]]] = field(default_factory=dict)


@dataclass
class _PutEntry:
    future: Future[list[int]]
    state: PutState = PutState.OPEN
    owner_execution_id: int | None = None


class StorePutRegistry:
    """Coordinates one native put session per canonical key in a Worker."""

    def __init__(self, backend: LayerwiseStoreBackend, replicate_config=None):
        self.backend = backend
        self.replicate_config = replicate_config
        self._entries: dict[str, _PutEntry] = {}
        self._lock = threading.RLock()

    def claim_execution(
        self,
        execution_id: int,
        ordered_bindings: list[PutBinding],
    ) -> PutClaim:
        """Claim keys in scheduler order and open at most one batch session."""

        unique: list[PutBinding] = []
        claim = PutClaim()
        with self._lock:
            seen: set[str] = set()
            for binding in ordered_bindings:
                if binding.key in seen:
                    claim.followed_keys.append(binding.key)
                    continue
                seen.add(binding.key)
                existing = self._entries.get(binding.key)
                if existing is not None and existing.state in {
                    PutState.ERROR,
                    PutState.REVOKED,
                }:
                    # Failed sessions are terminal for their execution only.
                    # Do not turn a later scheduler retry into a follower of a
                    # stale failed future.
                    self._entries.pop(binding.key, None)
                    existing = None
                if existing is not None and existing.state in {
                    PutState.END_FAILED,
                    PutState.CLEANUP_FAILED,
                }:
                    # Mooncake retains the native put session when put_end or
                    # put_revoke fails. Do not open a second session for the
                    # same canonical key until native cleanup is confirmed.
                    claim.failed_keys.append(binding.key)
                    claim.commit_futures[binding.key] = existing.future
                    continue
                if existing is not None:
                    claim.followed_keys.append(binding.key)
                    claim.commit_futures[binding.key] = existing.future
                    continue
                unique.append(binding)

            if unique:
                keys = [binding.key for binding in unique]
                try:
                    result = self.backend.batch_put_session_start(
                        keys,
                        [binding.object_size for binding in unique],
                        self.replicate_config,
                    )
                    if len(result) != len(keys):
                        raise RuntimeError("Mooncake put_start returned an invalid result length")
                except BaseException as error:
                    for binding in unique:
                        future: Future[list[int]] = Future()
                        future.set_exception(error)
                        self._entries[binding.key] = _PutEntry(
                            future=future,
                            state=PutState.ERROR,
                            owner_execution_id=execution_id,
                        )
                        claim.failed_keys.append(binding.key)
                        claim.commit_futures[binding.key] = future
                else:
                    for binding, code in zip(unique, result, strict=True):
                        if int(code) == MOONCAKE_OBJECT_ALREADY_EXISTS:
                            # Another (cross-process) writer owns this
                            # canonical key. The chunk hash guarantees the
                            # object content is equivalent, so yield write
                            # ownership: no local entry, no future, no put.
                            # Native put_start failed keys are not registered
                            # client-side (real_client), and master metadata
                            # belongs to the winner, so nothing leaks here.
                            # Visibility for later chunks is resolved by the
                            # existing stateless exists() triage in
                            # ``KVPoolWorker._start_load_range_request``; if
                            # the owner has not committed by then, the request
                            # fails through the existing history-incomplete
                            # path and a scheduler retry heals it.
                            claim.yielded_keys.append(binding.key)
                            continue
                        future = Future()
                        entry = _PutEntry(
                            future=future,
                            state=PutState.OPEN if code == 0 else PutState.ERROR,
                            owner_execution_id=execution_id,
                        )
                        self._entries[binding.key] = entry
                        claim.commit_futures[binding.key] = future
                        if code == 0:
                            claim.owned_keys.append(binding.key)
                        else:
                            future.set_exception(
                                StoreCommitError(
                                    f"put_start failed for {binding.key}: {code}"
                                )
                            )
                            claim.failed_keys.append(binding.key)
            for binding in ordered_bindings:
                entry = self._entries.get(binding.key)
                if entry is not None:
                    claim.commit_futures.setdefault(binding.key, entry.future)
        return claim

    def publish_commit_result(
        self,
        keys: list[str],
        result_or_error: list[int] | BaseException,
    ) -> None:
        with self._lock:
            if isinstance(result_or_error, BaseException):
                for key in keys:
                    entry = self._entries.get(key)
                    if entry is None or entry.future.done():
                        continue
                    entry.state = PutState.ERROR
                    entry.future.set_exception(result_or_error)
                return
            if len(keys) != len(result_or_error):
                error = RuntimeError("Mooncake put_end returned an invalid result length")
                return self.publish_commit_result(keys, error)
            for key, code in zip(keys, result_or_error, strict=True):
                entry = self._entries.get(key)
                if entry is None or entry.future.done():
                    continue
                if code == 0:
                    entry.state = PutState.COMMITTED
                    entry.future.set_result([code])
                else:
                    entry.state = PutState.END_FAILED
                    entry.future.set_exception(
                        StoreCommitError(f"put_end failed for {key}: {code}")
                    )

    def wait_followed(self, keys: list[str], timeout: float | None = None) -> None:
        for key in keys:
            entry = self._entries[key]
            entry.future.result(timeout=timeout)

    def mark_native_state(self, keys: list[str], state: PutState) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.state = state

    def publish_revoke_result(
        self,
        keys: list[str],
        result_or_error: list[int] | BaseException,
        terminal_error: BaseException,
    ) -> None:
        with self._lock:
            if isinstance(result_or_error, BaseException):
                for key in keys:
                    entry = self._entries.get(key)
                    if entry is None:
                        continue
                    entry.state = PutState.CLEANUP_FAILED
                    if not entry.future.done():
                        entry.future.set_exception(terminal_error)
                return
            if len(keys) != len(result_or_error):
                error = RuntimeError("Mooncake put_revoke returned an invalid result length")
                return self.publish_revoke_result(keys, error, terminal_error)
            for key, code in zip(keys, result_or_error, strict=True):
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.state = PutState.REVOKED if code == 0 else PutState.CLEANUP_FAILED
                if not entry.future.done():
                    entry.future.set_exception(terminal_error)

    def forget(self, keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if (
                    entry is not None
                    and entry.future.done()
                    and entry.state
                    in {
                        PutState.COMMITTED,
                        PutState.REVOKED,
                        PutState.ERROR,
                    }
                ):
                    self._entries.pop(key, None)


@dataclass
class _ReadEntry:
    references: set[tuple[str, int]] = field(default_factory=set)
    inflight_ranges: int = 0


class StoreReadLeaseRegistry:
    """Reference-counts complete objects shared by requests in one client."""

    def __init__(self, backend: LayerwiseStoreBackend):
        self.backend = backend
        self._entries: dict[str, _ReadEntry] = {}
        self._lock = threading.RLock()

    def _rollback_unreferenced_starts(
        self,
        keys: list[str],
        result: list[int],
        candidates: set[str],
    ) -> None:
        started = [
            key
            for key, code in zip(keys, result, strict=False)
            if code == 0 and key in candidates
        ]
        if not started:
            return
        try:
            code = self.backend.batch_get_session_end(started)
        except BaseException as error:
            raise StoreCommitError(
                f"Failed to roll back Store get sessions: {started!r}"
            ) from error
        if code != 0:
            raise StoreCommitError(
                f"Failed to roll back Store get sessions: {started!r}, {code}"
            )

    def acquire(self, owner: tuple[str, int], keys: list[str]) -> None:
        unique = list(dict.fromkeys(keys))
        to_start: list[str] = []
        with self._lock:
            for key in unique:
                entry = self._entries.setdefault(key, _ReadEntry())
                if not entry.references:
                    to_start.append(key)
            if to_start:
                try:
                    result = self.backend.batch_get_session_start(to_start)
                except BaseException:
                    for key in to_start:
                        if not self._entries[key].references:
                            self._entries.pop(key, None)
                    raise
                if len(result) != len(to_start) or any(
                    code != 0 for code in result
                ):
                    cleanup_error: BaseException | None = None
                    try:
                        self._rollback_unreferenced_starts(
                            to_start, result, set(to_start)
                        )
                    except BaseException as error:
                        cleanup_error = error
                    finally:
                        for key in to_start:
                            if not self._entries[key].references:
                                self._entries.pop(key, None)
                    if cleanup_error is not None:
                        raise StoreCommitError(
                            "batch_get_session_start failed and rollback "
                            f"also failed: {to_start!r}, {result!r}"
                        ) from cleanup_error
                    raise StoreCommitError(
                        "batch_get_session_start failed: "
                        f"{to_start!r}, {result!r}"
                    )
            for key in unique:
                self._entries[key].references.add(owner)

    def refresh(self, owner: tuple[str, int], keys: list[str]) -> None:
        """Refresh the native get sessions for an owner's complete history."""
        unique = list(dict.fromkeys(keys))
        if not unique:
            return
        with self._lock:
            created: list[str] = []
            for key in unique:
                entry = self._entries.get(key)
                if entry is None:
                    entry = _ReadEntry()
                    self._entries[key] = entry
                    created.append(key)
                elif entry.inflight_ranges:
                    raise RuntimeError(
                        f"Cannot refresh Store key {key!r} with an in-flight range"
                    )
            try:
                result = self.backend.batch_get_session_start(unique)
            except BaseException:
                for key in created:
                    if not self._entries[key].references:
                        self._entries.pop(key, None)
                raise
            if len(result) != len(unique) or any(
                code != 0 for code in result
            ):
                cleanup_error: BaseException | None = None
                try:
                    self._rollback_unreferenced_starts(
                        unique, result, set(created)
                    )
                except BaseException as error:
                    cleanup_error = error
                finally:
                    for key in created:
                        if not self._entries[key].references:
                            self._entries.pop(key, None)
                if cleanup_error is not None:
                    raise StoreCommitError(
                        "batch_get_session_start refresh failed and rollback "
                        f"also failed: {unique!r}, {result!r}"
                    ) from cleanup_error
                raise StoreCommitError(
                    f"batch_get_session_start failed: {unique!r}, {result!r}"
                )
            for key in unique:
                self._entries[key].references.add(owner)

    def begin_range(self, owner: tuple[str, int], keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None or owner not in entry.references:
                    raise RuntimeError(f"Owner {owner!r} has no Store lease for {key!r}")
                entry.inflight_ranges += 1

    def end_range(self, keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries[key]
                entry.inflight_ranges -= 1
                if entry.inflight_ranges < 0:
                    raise RuntimeError("Store range completion counted twice")

    def release(self, owner: tuple[str, int], keys: list[str]) -> None:
        to_end: list[str] = []
        with self._lock:
            for key in list(dict.fromkeys(keys)):
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.references.discard(owner)
                if not entry.references:
                    if entry.inflight_ranges:
                        raise RuntimeError(
                            f"Cannot release Store key {key!r} with an in-flight range"
                        )
                    to_end.append(key)
            if to_end:
                try:
                    code = self.backend.batch_get_session_end(to_end)
                except BaseException as error:
                    # The native read session is no longer safe to reuse.
                    # Drop local entries even when the backend reports an end
                    # failure, then propagate the failure to the request.
                    for key in to_end:
                        self._entries.pop(key, None)
                    raise StoreCommitError(
                        f"batch_get_session_end failed: {to_end!r}"
                    ) from error
                for key in to_end:
                    self._entries.pop(key, None)
                if code != 0:
                    raise StoreCommitError(f"batch_get_session_end failed: {to_end!r}, {code}")


class RequestStoreLease:
    def __init__(self, owner: tuple[str, int], registry: StoreReadLeaseRegistry):
        self.owner = owner
        self.registry = registry
        self.history_keys: set[str] = set()
        self.closed = False

    def ensure_history(self, keys: list[str]) -> None:
        if self.closed:
            raise RuntimeError("Request Store lease is closed")
        missing = sorted(set(keys) - self.history_keys)
        if missing:
            self.registry.acquire(self.owner, missing)
            self.history_keys.update(missing)

    def refresh_history(self, keys: list[str]) -> None:
        if self.closed:
            raise RuntimeError("Request Store lease is closed")
        history = sorted(set(keys))
        if history:
            self.registry.refresh(self.owner, history)
            self.history_keys.update(history)

    def load_layer(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_src_offsets: list[list[int]],
    ) -> list[int]:
        self.registry.begin_range(self.owner, keys)
        try:
            return self.registry.backend.batch_get_into_multi_buffer_ranges(
                keys,
                all_buffer_ptrs,
                all_sizes,
                all_src_offsets,
            )
        finally:
            self.registry.end_range(keys)

    def release(self) -> None:
        if self.closed:
            return
        try:
            self.registry.release(self.owner, sorted(self.history_keys))
        finally:
            # A failed batch_get_session_end is terminal for this lease; retaining the
            # owner keys would make a retry call the native end operation twice.
            self.history_keys.clear()
            self.closed = True


class ChunkStoreSession:
    """Current-chunk writer; it never closes the request history lease."""

    def __init__(
        self,
        request_chunk: RequestChunkKey,
        backend: LayerwiseStoreBackend,
        request_lease: RequestStoreLease,
        put_registry: StorePutRegistry,
    ):
        self.request_chunk = request_chunk
        self.backend = backend
        self.request_lease = request_lease
        self.put_registry = put_registry
        self.put_keys: list[str] = []
        self.followed_put_keys: list[str] = []
        self.put_state: dict[str, PutState] = {}
        self.put_end_attempted = False
        self.closed = False

    def start(
        self,
        history_keys: list[str],
        claim: PutClaim,
    ) -> None:
        # Bind the native put ownership before acquiring history. If history
        # setup fails, the caller can still revoke every successfully opened
        # put session through revoke_uncommitted().
        self.put_keys = list(dict.fromkeys(claim.owned_keys))
        self.followed_put_keys = list(dict.fromkeys(claim.followed_keys))
        self.put_state = {key: PutState.OPEN for key in self.put_keys}
        self.request_lease.ensure_history(history_keys)
        if claim.failed_keys:
            details: dict[str, str] = {}
            for key in claim.failed_keys:
                future = claim.commit_futures.get(key)
                error = future.exception() if future is not None and future.done() else None
                details[key] = repr(error) if error is not None else "unknown error"
            raise StoreCommitError(f"put_start failed: {details!r}")

    def save_layer(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_dst_offsets: list[list[int]],
        ready_event=None,
    ) -> list[int]:
        if self.closed or self.put_end_attempted:
            raise RuntimeError("Cannot save a closed Store chunk")
        if ready_event is not None:
            # This method runs in a background CPU thread and invokes the
            # Mooncake range API directly with raw buffer addresses. A
            # stream-local Event.wait() would not order that CPU API call
            # after the producer stream. Synchronize the event here so the
            # source KV is complete before Mooncake starts reading it.
            ready_event.synchronize()
        result = self.backend.batch_put_from_multi_buffer_ranges(
            keys,
            all_buffer_ptrs,
            all_sizes,
            all_dst_offsets,
        )
        expected_bytes = tuple(sum(row) for row in all_sizes)
        if len(result) != len(keys) or any(
            int(code) != expected
            for code, expected in zip(result, expected_bytes, strict=False)
        ):
            raise StoreCommitError(
                f"Store range put failed: result={result!r}, "
                f"expected={expected_bytes!r}"
            )
        return result

    def commit_owned(self) -> list[int]:
        if self.put_end_attempted:
            raise RuntimeError("Store put_end was already attempted")
        self.put_end_attempted = True
        if not self.put_keys:
            self.closed = True
            return []
        try:
            result = self.backend.batch_put_session_end(self.put_keys)
        except BaseException as error:
            self.put_registry.publish_commit_result(self.put_keys, error)
            for key in self.put_keys:
                self.put_state[key] = PutState.END_FAILED
            self.closed = True
            try:
                self._reconcile_unknown_end(error)
            except BaseException as cleanup_error:
                raise StoreCommitError(
                    f"Store put_end failed with {error!r}; cleanup failed with "
                    f"{cleanup_error!r}"
                ) from error
            raise
        if len(result) != len(self.put_keys):
            error = StoreCommitError(
                "Mooncake put_end returned an invalid result length: "
                f"expected={len(self.put_keys)}, actual={len(result)}"
            )
            self.put_registry.publish_commit_result(self.put_keys, error)
            for key in self.put_keys:
                self.put_state[key] = PutState.END_FAILED
            self.closed = True
            try:
                self._reconcile_unknown_end(error)
            except BaseException as cleanup_error:
                raise StoreCommitError(
                    f"{error}; cleanup failed with {cleanup_error!r}"
                ) from error
            raise error
        self.put_registry.publish_commit_result(self.put_keys, result)
        for key, code in zip(self.put_keys, result, strict=True):
            self.put_state[key] = PutState.COMMITTED if code == 0 else PutState.END_FAILED
        self.closed = True
        failed_keys = [
            key
            for key, code in zip(self.put_keys, result, strict=True)
            if code != 0
        ]
        if failed_keys:
            error = StoreCommitError(f"Store put_end failed: {result!r}")
            try:
                self._revoke_keys(failed_keys, error)
            except BaseException as cleanup_error:
                raise StoreCommitError(
                    f"{error}; cleanup failed with {cleanup_error!r}"
                ) from error
            raise error
        return result

    def _reconcile_unknown_end(self, terminal_error: BaseException) -> None:
        """Resolve native sessions after an ambiguous put_end result."""

        try:
            exists = self.backend.exists(self.put_keys)
        except BaseException as error:
            self.put_registry.mark_native_state(
                self.put_keys, PutState.CLEANUP_FAILED
            )
            for key in self.put_keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise StoreCommitError(
                f"Store complete lookup failed after put_end: {error!r}"
            ) from error
        if len(exists) != len(self.put_keys) or any(code not in (0, 1) for code in exists):
            self.put_registry.mark_native_state(
                self.put_keys, PutState.CLEANUP_FAILED
            )
            for key in self.put_keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise StoreCommitError(
                "Store complete lookup returned an invalid result after "
                f"put_end: {exists!r}"
            )

        complete_keys = [
            key
            for key, is_complete in zip(self.put_keys, exists, strict=True)
            if is_complete == 1
        ]
        pending_keys = [
            key
            for key, is_complete in zip(self.put_keys, exists, strict=True)
            if is_complete == 0
        ]
        self.put_registry.mark_native_state(complete_keys, PutState.COMMITTED)
        for key in complete_keys:
            self.put_state[key] = PutState.COMMITTED
        self._revoke_keys(pending_keys, terminal_error)

    def _revoke_keys(
        self,
        keys: list[str],
        terminal_error: BaseException,
    ) -> None:
        if not keys:
            return
        try:
            result = self.backend.batch_put_session_revoke(keys)
        except BaseException as error:
            self.put_registry.publish_revoke_result(keys, error, terminal_error)
            for key in keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise
        self.put_registry.publish_revoke_result(keys, result, terminal_error)
        if len(result) != len(keys):
            for key in keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise StoreCommitError(
                "Mooncake put_revoke returned an invalid result length: "
                f"expected={len(keys)}, actual={len(result)}"
            )
        failed: dict[str, int] = {}
        for key, code in zip(keys, result, strict=True):
            if code == 0:
                self.put_state[key] = PutState.REVOKED
            else:
                self.put_state[key] = PutState.CLEANUP_FAILED
                failed[key] = int(code)
        if failed:
            raise StoreCommitError(f"Store put_revoke failed: {failed!r}")

    def finalize_followed(self, has_more_prefill_chunks: bool) -> None:
        self.put_registry.wait_followed(self.followed_put_keys)
        all_keys = list(dict.fromkeys(self.put_keys + self.followed_put_keys))
        if has_more_prefill_chunks:
            self.request_lease.refresh_history(
                list(self.request_lease.history_keys) + all_keys
            )

    def revoke_uncommitted(self) -> None:
        if not self.put_keys:
            return
        keys = [
            key
            for key in self.put_keys
            if self.put_state.get(key)
            in {
                PutState.OPEN,
                PutState.END_FAILED,
                PutState.CLEANUP_FAILED,
            }
        ]
        if not keys:
            return
        self._revoke_keys(
            keys,
            StoreCommitError("Store put was revoked before commit"),
        )
