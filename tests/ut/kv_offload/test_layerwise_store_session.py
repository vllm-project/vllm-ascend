# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_session import (
    ChunkStoreSession,
    PutBinding,
    PutState,
    RequestStoreLease,
    StoreCommitError,
    StorePutRegistry,
    StoreReadLeaseRegistry,
)
from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import RequestChunkKey


class FakeBackend:
    def __init__(self):
        self.calls = Counter()
        self.get_start_keys = []
        self.get_end_keys = []
        self.get_start_result = None
        self.exists_result = None
        self.exists_error = None
        self.put_start_result = None
        self.put_end_result = None
        self.put_end_error = None
        self.put_revoke_result = None
        self.put_revoke_error = None
        self.put_range_result = None
        self.put_revoke_keys = []

    def exists(self, keys):
        self.calls["exists"] += 1
        if self.exists_error is not None:
            raise self.exists_error
        if self.exists_result is not None:
            return list(self.exists_result)
        return [0] * len(keys)

    def batch_put_session_start(self, keys, sizes, config=None):
        del sizes, config
        self.calls["put_start"] += 1
        return self.put_start_result or [0] * len(keys)

    def batch_put_from_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        del ptrs, offsets
        self.calls["put_range"] += 1
        return self.put_range_result or [sum(row) for row in sizes]

    def batch_put_session_end(self, keys):
        self.calls["put_end"] += 1
        if self.put_end_error is not None:
            raise self.put_end_error
        if self.put_end_result is not None:
            return list(self.put_end_result)
        return [0] * len(keys)

    def batch_put_session_revoke(self, keys):
        self.calls["put_revoke"] += 1
        self.put_revoke_keys.append(list(keys))
        if self.put_revoke_error is not None:
            raise self.put_revoke_error
        if self.put_revoke_result is not None:
            return list(self.put_revoke_result)
        return [0] * len(keys)

    def batch_get_session_start(self, keys):
        self.calls["get_start"] += 1
        self.get_start_keys.append(list(keys))
        if self.get_start_result is not None:
            return list(self.get_start_result)
        return [0] * len(keys)

    def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        del ptrs, offsets
        self.calls["get_range"] += 1
        return [sum(row) for row in sizes]

    def batch_get_session_end(self, keys):
        self.calls["get_end"] += 1
        self.get_end_keys.append(list(keys))
        return 0


def test_same_execution_key_has_one_writer_and_follower():
    backend = FakeBackend()
    registry = StorePutRegistry(backend)
    claim = registry.claim_execution(
        1,
        [PutBinding("same", 128), PutBinding("same", 128)],
    )
    assert backend.calls["put_start"] == 1
    assert claim.owned_keys == ["same"]
    assert claim.followed_keys == ["same"]
    registry.publish_commit_result(["same"], [0])
    registry.wait_followed(claim.followed_keys, timeout=0)


def test_owner_failure_resolves_follower_future():
    backend = FakeBackend()
    registry = StorePutRegistry(backend)
    claim = registry.claim_execution(
        1,
        [PutBinding("same", 128), PutBinding("same", 128)],
    )
    registry.publish_commit_result(["same"], RuntimeError("commit failed"))
    with pytest.raises(RuntimeError, match="commit failed"):
        registry.wait_followed(claim.followed_keys, timeout=0)


def test_put_start_failure_reports_backend_code():
    backend = FakeBackend()
    backend.put_start_result = [-2]
    registry = StorePutRegistry(backend)
    claim = registry.claim_execution(1, [PutBinding("duplicate", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0),
        backend,
        RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend)),
        registry,
    )

    with pytest.raises(StoreCommitError, match=r"duplicate.*-2"):
        session.start([], claim)


def test_request_lease_is_reference_counted_across_requests():
    backend = FakeBackend()
    registry = StoreReadLeaseRegistry(backend)
    first = RequestStoreLease(("a", 0), registry)
    second = RequestStoreLease(("b", 0), registry)
    first.ensure_history(["prefix"])
    second.ensure_history(["prefix"])
    assert backend.calls["get_start"] == 1
    first.release()
    assert backend.calls["get_end"] == 0
    second.release()
    assert backend.calls["get_end"] == 1


def test_chunk_commit_then_retains_new_history_for_next_chunk():
    backend = FakeBackend()
    put_registry = StorePutRegistry(backend)
    read_registry = StoreReadLeaseRegistry(backend)
    lease = RequestStoreLease(("request", 0), read_registry)
    claim = put_registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0),
        backend,
        lease,
        put_registry,
    )
    session.start([], claim)
    session.save_layer(["new"], [[1000]], [[64]], [[0]])
    session.commit_owned()
    session.finalize_followed(has_more_prefill_chunks=True)
    assert lease.history_keys == {"new"}
    assert backend.calls["put_end"] == 1
    assert backend.calls["get_start"] == 1


def test_chunk_commit_refreshes_complete_request_history():
    backend = FakeBackend()
    put_registry = StorePutRegistry(backend)
    read_registry = StoreReadLeaseRegistry(backend)
    lease = RequestStoreLease(("request", 0), read_registry)
    lease.ensure_history(["old-0", "old-1"])
    claim = put_registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 1),
        backend,
        lease,
        put_registry,
    )
    session.start(sorted(lease.history_keys), claim)
    session.save_layer(["new"], [[1000]], [[64]], [[0]])
    session.commit_owned()
    session.finalize_followed(has_more_prefill_chunks=True)

    assert lease.history_keys == {"old-0", "old-1", "new"}
    assert backend.get_start_keys == [
        ["old-0", "old-1"],
        ["new", "old-0", "old-1"],
    ]


def test_request_lease_rejects_refresh_with_inflight_range():
    backend = FakeBackend()
    registry = StoreReadLeaseRegistry(backend)
    lease = RequestStoreLease(("request", 0), registry)
    lease.ensure_history(["history"])
    registry.begin_range(lease.owner, ["history"])

    with pytest.raises(RuntimeError, match="in-flight range"):
        lease.refresh_history(["history"])

    registry.end_range(["history"])


def test_get_start_partial_success_rolls_back_only_started_keys():
    backend = FakeBackend()
    backend.get_start_result = [0, -1]
    registry = StoreReadLeaseRegistry(backend)

    with pytest.raises(StoreCommitError, match="batch_get_session_start failed"):
        registry.acquire(("request", 0), ["started", "failed"])

    assert backend.get_end_keys == [["started"]]
    assert registry._entries == {}

    # A later retry must not inherit either half of the failed native batch.
    backend.get_start_result = None
    registry.acquire(("request", 0), ["started", "failed"])
    assert set(registry._entries) == {"started", "failed"}


def test_refresh_partial_success_preserves_existing_lease_and_rolls_back_new_key():
    backend = FakeBackend()
    registry = StoreReadLeaseRegistry(backend)
    owner = ("request", 0)
    registry.acquire(owner, ["existing"])
    backend.get_start_result = [0, 0, -1]

    with pytest.raises(StoreCommitError, match="batch_get_session_start failed"):
        registry.refresh(owner, ["existing", "new-started", "new-failed"])

    # Ending the refreshed existing key would invalidate its original owner.
    # Roll back only newly opened, previously unreferenced keys.
    assert backend.get_end_keys == [["new-started"]]
    assert set(registry._entries) == {"existing"}
    assert registry._entries["existing"].references == {owner}


def test_chunk_start_can_revoke_put_when_history_initialization_fails():
    backend = FakeBackend()
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    backend.get_start_result = [-1]
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )

    with pytest.raises(StoreCommitError, match="batch_get_session_start failed"):
        session.start(["history"], claim)
    session.revoke_uncommitted()

    assert backend.calls["put_revoke"] == 1


def test_range_put_synchronizes_ready_event_before_reading_source():
    backend = FakeBackend()
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)

    call_order = []
    ready_event = MagicMock()
    ready_event.synchronize.side_effect = lambda: call_order.append("synchronize")
    original_put_range = backend.batch_put_from_multi_buffer_ranges

    def traced_put_range(keys, ptrs, sizes, offsets):
        call_order.append("put_range")
        return original_put_range(keys, ptrs, sizes, offsets)

    backend.batch_put_from_multi_buffer_ranges = traced_put_range
    session.save_layer(
        ["new"],
        [[1000]],
        [[64]],
        [[0]],
        ready_event=ready_event,
    )

    assert call_order == ["synchronize", "put_range"]
    ready_event.synchronize.assert_called_once_with()
    ready_event.wait.assert_not_called()


def test_range_put_rejects_unexpected_positive_byte_count():
    backend = FakeBackend()
    backend.put_range_result = [1]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)
    with pytest.raises(StoreCommitError, match="expected"):
        session.save_layer(["new"], [[1000]], [[64]], [[0]])


def test_put_end_failure_revokes_retained_native_session():
    backend = FakeBackend()
    backend.put_end_result = [-7]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)
    with pytest.raises(StoreCommitError):
        session.commit_owned()
    session.revoke_uncommitted()
    assert backend.put_revoke_keys == [["new"]]
    assert session.put_state == {"new": PutState.REVOKED}


def test_put_end_partial_failure_revokes_only_failed_key():
    backend = FakeBackend()
    backend.put_end_result = [0, -7]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(
        1,
        [PutBinding("complete", 128), PutBinding("failed", 128)],
    )
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)

    with pytest.raises(StoreCommitError, match="put_end failed"):
        session.commit_owned()

    assert backend.put_revoke_keys == [["failed"]]
    assert session.put_state == {
        "complete": PutState.COMMITTED,
        "failed": PutState.REVOKED,
    }
    assert claim.commit_futures["complete"].result() == [0]
    with pytest.raises(StoreCommitError, match="failed"):
        claim.commit_futures["failed"].result()


def test_put_end_exception_reconciles_complete_keys_before_revoke():
    backend = FakeBackend()
    backend.put_end_error = RuntimeError("end rpc failed")
    backend.exists_result = [1, 0]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(
        1,
        [PutBinding("complete", 128), PutBinding("pending", 128)],
    )
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)

    with pytest.raises(RuntimeError, match="end rpc failed"):
        session.commit_owned()

    assert backend.put_revoke_keys == [["pending"]]
    assert session.put_state == {
        "complete": PutState.COMMITTED,
        "pending": PutState.REVOKED,
    }


def test_put_end_invalid_result_length_reconciles_native_sessions():
    backend = FakeBackend()
    backend.put_end_result = [0]
    backend.exists_result = [1, 0]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(
        1,
        [PutBinding("complete", 128), PutBinding("pending", 128)],
    )
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)

    with pytest.raises(StoreCommitError, match="invalid result length"):
        session.commit_owned()

    assert backend.calls["exists"] == 1
    assert backend.put_revoke_keys == [["pending"]]


def test_put_revoke_failure_blocks_restart_until_cleanup_succeeds():
    backend = FakeBackend()
    backend.put_end_result = [-7]
    backend.put_revoke_result = [-9]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)

    with pytest.raises(StoreCommitError, match="cleanup failed"):
        session.commit_owned()

    registry.forget(["new"])
    assert registry._entries["new"].state is PutState.CLEANUP_FAILED
    retry = registry.claim_execution(2, [PutBinding("new", 128)])
    assert retry.failed_keys == ["new"]
    assert backend.calls["put_start"] == 1

    backend.put_revoke_result = [0]
    session.revoke_uncommitted()
    registry.forget(["new"])
    recovered = registry.claim_execution(3, [PutBinding("new", 128)])
    assert recovered.owned_keys == ["new"]
    assert backend.calls["put_start"] == 2
