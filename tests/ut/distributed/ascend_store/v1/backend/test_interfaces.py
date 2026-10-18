"""Object-session ownership and cleanup, without Worker plans or Mapping inputs."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    GVARegion,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.backend import GVA_REVOKE_UNCONFIRMED
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.sessions import MemcacheSessions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake.sessions import MooncakeSessions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.results import integer_results


def inject_backend_failure(method, native_result):
    # Match the real adapter contract: malformed SDK results become an exception
    # before Session can update any ownership record.
    def call(keys, *_args):
        if isinstance(native_result, Exception):
            raise native_result
        return integer_results("injected SDK result", keys, native_result)

    method.side_effect = call


@pytest.fixture(params=("key_range", "gva"))
def owner(request):
    backend = MagicMock()
    if request.param == "gva":
        backend.allocate_gva.side_effect = lambda keys, sizes: tuple(1000 + index * 64 for index in range(len(keys)))
        backend.add_gva_leases.side_effect = lambda keys: (0,) * len(keys)
        backend.query_gva_regions.side_effect = lambda keys: tuple(
            GVARegion(2000 + index * 64, 64) for index in range(len(keys))
        )
        backend.publish_gva.side_effect = lambda keys: (0,) * len(keys)
        backend.revoke_gva.side_effect = lambda keys: (0,) * len(keys)
        sessions = MemcacheSessions(backend)
        acquire = backend.add_gva_leases
        release = backend.remove_gva_leases
        allocate = backend.allocate_gva
        publish = backend.publish_gva
        revoke = backend.revoke_gva
    else:
        backend.start_key_range_load.side_effect = lambda keys: (0,) * len(keys)
        backend.start_key_range_store.side_effect = lambda keys, sizes: (0,) * len(keys)
        backend.commit_key_range_store.side_effect = lambda keys: (0,) * len(keys)
        backend.revoke_key_range_store.side_effect = lambda keys: (0,) * len(keys)
        sessions = MooncakeSessions(backend)
        acquire = backend.start_key_range_load
        release = backend.finish_key_range_load
        allocate = backend.start_key_range_store
        publish = backend.commit_key_range_store
        revoke = backend.revoke_key_range_store
    return sessions, backend, acquire, release, allocate, publish, revoke


def test_partial_read_open_releases_only_owned_keys_and_can_retry(owner):
    sessions, _, acquire, release, *_ = owner
    acquire.side_effect = None
    acquire.return_value = (0, -7)
    assert sessions.start_load(["owned", "rejected"], [64, 64]) == (0, -7)
    release.side_effect = [RuntimeError("release failed"), None]

    with pytest.raises(RuntimeError, match="release failed"):
        sessions.finish_load(["owned", "rejected", "owned"])
    assert sessions.has_load_sessions
    if isinstance(sessions, MemcacheSessions):
        with pytest.raises(RuntimeError, match="unavailable"):
            sessions.load_regions(("owned",), 64)
    sessions.finish_load()
    sessions.finish_load()
    assert not sessions.has_load_sessions
    assert release.call_args_list == [call(["owned"]), call(["owned"])]


@pytest.mark.parametrize("failure", (RuntimeError("open outcome unknown"), (0,), (0, True)))
def test_exceptional_read_open_keeps_attempts_for_cleanup(owner, failure):
    sessions, _, acquire, release, *_ = owner
    inject_backend_failure(acquire, failure)

    with pytest.raises(RuntimeError):
        sessions.start_load(["first", "second"], [64, 64])
    assert sessions.has_load_sessions
    sessions.finish_load()
    release.assert_called_once_with(["first", "second"])


def test_partial_store_open_never_revokes_rejected_keys(owner):
    sessions, _, _, _, allocate, publish, revoke = owner
    allocate.side_effect = None
    allocate.return_value = (1000, 0) if isinstance(sessions, MemcacheSessions) else (0, -7)
    assert sessions.start_store(["owned", "rejected"], [64, 64])[0] == 0
    assert sessions.pending_store_keys == ("owned",)

    assert sessions.revoke_store(["owned", "rejected", "owned"]) == (0, 0, 0)
    assert not sessions.pending_store_keys
    revoke.assert_called_once_with(["owned"])
    publish.assert_not_called()


@pytest.mark.parametrize("failure", (RuntimeError("allocation unknown"), (1000,), (1000, True)))
def test_exceptional_store_open_keeps_attempts_but_cannot_publish(owner, failure):
    sessions, _, _, _, allocate, publish, revoke = owner
    inject_backend_failure(allocate, failure)

    with pytest.raises(RuntimeError):
        sessions.start_store(["first", "second"], [64, 64])
    assert sessions.pending_store_keys == ("first", "second")
    with pytest.raises(RuntimeError, match="no confirmed owning"):
        sessions.commit_store(["first"])
    publish.assert_not_called()
    sessions.revoke_store(["first", "second"])
    revoke.assert_called_once_with(["first", "second"])
    assert not sessions.pending_store_keys


def test_failed_revoke_retains_ownership_blocks_reopen_and_is_retryable(owner):
    sessions, _, _, _, allocate, publish, revoke = owner
    assert sessions.start_store(["key"], [64]) == (0,)
    revoke.side_effect = [(GVA_REVOKE_UNCONFIRMED,), (0,)]

    assert sessions.revoke_store(["key"]) == (GVA_REVOKE_UNCONFIRMED,)
    assert sessions.pending_store_keys == ("key",)
    with pytest.raises(RuntimeError, match="have not ended"):
        sessions.start_store(["key"], [64])
    with pytest.raises(RuntimeError, match="no confirmed owning"):
        sessions.commit_store(["key"])
    allocate.assert_called_once_with(["key"], [64])
    publish.assert_not_called()
    if isinstance(sessions, MemcacheSessions):
        with pytest.raises(RuntimeError, match="unavailable"):
            sessions.store_regions(("key",), 64)
    assert sessions.revoke_store(["key"]) == (0,)
    assert sessions.revoke_store(["key"]) == (0,)
    assert not sessions.pending_store_keys
    assert revoke.call_count == 2


def test_partial_commit_cleans_only_failed_writes(owner):
    sessions, _, _, _, _, publish, revoke = owner
    sessions.start_store(["published", "failed"], [64, 64])
    publish.side_effect = None
    publish.return_value = (0, -8)

    assert sessions.commit_store(["published", "failed"]) == (0, -8)
    assert sessions.pending_store_keys == ("failed",)
    assert sessions.revoke_store(["published", "failed"]) == (0, 0)
    revoke.assert_called_once_with(["failed"])


@pytest.mark.parametrize("stage", ("publish", "revoke"))
@pytest.mark.parametrize("failure", (RuntimeError("native failure"), None, (), (True,), (0, 0)))
def test_failed_or_malformed_finalization_retains_ownership(owner, stage, failure):
    sessions, _, _, _, _, publish, revoke = owner
    sessions.start_store(["key"], [64])
    native = publish if stage == "publish" else revoke
    operation = sessions.commit_store if stage == "publish" else sessions.revoke_store
    inject_backend_failure(native, failure)

    with pytest.raises(RuntimeError):
        operation(["key"])
    assert sessions.pending_store_keys == ("key",)
    revoke.side_effect = None
    revoke.return_value = (0,)
    sessions.close()
    assert not sessions.pending_store_keys


@pytest.mark.parametrize("direction", ("load", "store"))
@pytest.mark.parametrize("keys,sizes", ((["a", "a"], [64, 64]), (["a"], []), (["a"], [0])))
def test_invalid_open_does_not_acquire_native_resources(owner, direction, keys, sizes):
    sessions, backend, *_ = owner
    backend.reset_mock()
    with pytest.raises(ValueError):
        getattr(sessions, f"start_{direction}")(keys, sizes)
    assert backend.mock_calls == []
    assert not sessions.has_load_sessions
    assert not sessions.pending_store_keys


def test_empty_sessions_and_repeated_close_do_not_call_native_operations(owner):
    sessions, backend, *_ = owner
    backend.reset_mock()
    assert sessions.start_load([], []) == ()
    assert sessions.start_store([], []) == ()
    assert sessions.commit_store([]) == ()
    assert sessions.revoke_store([]) == ()
    sessions.close()
    sessions.close()
    assert backend.mock_calls == []


def test_close_retains_unconfirmed_writes_and_retries_without_releasing_reads_twice(owner):
    sessions, _, _, release, _, _, revoke = owner
    sessions.start_load(["read"], [64])
    sessions.start_store(["write"], [64])
    revoke.side_effect = [(-8,), (0,)]

    with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
        sessions.close()
    assert not sessions.has_load_sessions
    assert sessions.pending_store_keys == ("write",)
    sessions.close()
    sessions.close()
    assert not sessions.pending_store_keys
    release.assert_called_once_with(["read"])
    assert revoke.call_args_list == [call(["write"]), call(["write"])]
