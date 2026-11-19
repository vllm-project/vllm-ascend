# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_io import MemcacheStoreLayoutIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import (
    PoolObject,
    PoolRange,
)


class MemoryBackend:
    def set_device(self):
        pass

    def __init__(self):
        self.objects = {}
        self.readable = set()
        self.leases = set()
        self.store = self
        self.fail_copy = False
        self.registered = {}

    def register_buffer(self, addresses, sizes):
        self.registered.update(zip(addresses, sizes, strict=True))

    def unregister_buffer(self, addresses, sizes):
        for address, size in zip(addresses, sizes, strict=True):
            assert self.registered.pop(address) == size

    def batch_is_readable(self, keys):
        return [key in self.readable for key in keys]

    def batch_alloc(self, keys, sizes, ttl):
        for key, size in zip(keys, sizes):
            self.objects.setdefault(key, ctypes.create_string_buffer(size))
        self.leases.update(keys)
        return [ctypes.addressof(self.objects[key]) for key in keys]

    def batch_add_lease(self, keys, ttl):
        self.leases.update(key for key in keys if key in self.readable)
        return [0 if key in self.readable else -1 for key in keys]

    def batch_get_key_info(self, keys, **kwargs):
        return [
            SimpleNamespace(
                size=lambda key=key: len(self.objects[key]),
                gva_list=lambda key=key: [ctypes.addressof(self.objects[key])],
            )
            for key in keys
        ]

    def batch_remove_lease(self, keys):
        self.leases.difference_update(keys)
        return 0

    def batch_write_finish(self, keys, results):
        self.readable.update(keys)
        self.leases.difference_update(keys)  # WRITE_OK consumes the write lease.
        return [0] * len(keys)

    def batch_copy(self, gvas, addresses, sizes, direction):
        if self.fail_copy:
            return -1
        for gva, address, size in zip(gvas, addresses, sizes):
            ctypes.memmove(gva if direction == 0 else address, address if direction == 0 else gva, size)
        return 0


def plan(buffer):
    address = ctypes.addressof(buffer)
    return PoolObject(
        "shared-key",
        8,
        1,
        (
            PoolRange("layer.0", address, 0, 4),
            PoolRange("layer.1", address + 4, 4, 4),
        ),
    )


@pytest.mark.parametrize("layerwise_write", [False, True])
@pytest.mark.parametrize("layerwise_read", [False, True])
def test_interoperable_modes_and_publication(layerwise_write, layerwise_read):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    session = io.begin([plan(source)], writing=True)
    for layers in ({"layer.0"}, {"layer.1"}) if layerwise_write else (None,):
        io.copy(session, layers)
        assert io.batch_is_exist(["shared-key"]) == [0]
    io.finish(session)
    assert not backend.leases
    target = ctypes.create_string_buffer(8)
    reader = MemcacheStoreLayoutIO(backend)
    session = reader.begin([plan(target)], writing=False)
    for layers in ({"layer.0"}, {"layer.1"}) if layerwise_read else (None,):
        reader.copy(session, layers)
    reader.finish(session)
    assert target.raw == source.raw
    assert not backend.leases


@pytest.mark.parametrize(
    "registration_split, bulk, sizes", [(False, False, [8]), (True, False, [4, 4]), (False, True, [4, 4])]
)
def test_read_copy_coalesces_only_contiguous_ranges_inside_one_registration(
    monkeypatch, registration_split, bulk, sizes
):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    target = ctypes.create_string_buffer(8)
    address = ctypes.addressof(target)
    io.set_coalesced_read_regions({address: 4, address + 4: 4} if registration_split else {address: 8})
    source_plan = plan(source)
    target_plan = replace(
        plan(target), ranges=(PoolRange("layer.0", address, 0, 4), PoolRange("layer.0", address + 4, 4, 4))
    )
    submit = backend.batch_copy
    native_sizes = []

    def record(gvas, local, lengths, direction):
        native_sizes.append(lengths)
        return submit(gvas, local, lengths, direction)

    monkeypatch.setattr(backend, "batch_copy", record)
    session = io.begin([source_plan], writing=True)
    io.copy(session)
    io.finish(session)
    session = io.begin([target_plan], writing=False)
    if not bulk:
        io.prepare_layer_reads(session, (frozenset({"layer.0"}),))
    io.copy(session, None if bulk else {"layer.0"})
    io.finish(session)
    assert native_sizes == [[4, 4], sizes]
    assert (session.logical_read_segments, session.submitted_read_segments) == ((0, 0) if bulk else (2, len(sizes)))
    assert target.raw == source.raw
    assert not backend.leases


def test_prepared_bundle_matches_original_native_vectors(monkeypatch):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    target = ctypes.create_string_buffer(8)
    address = ctypes.addressof(target)
    io.set_coalesced_read_regions({address: 8})
    writer = io.begin([plan(source)], writing=True)
    io.copy(writer)
    io.finish(writer)

    native = backend.batch_copy
    calls = []

    def record(*args):
        calls.append(args)
        return native(*args)

    monkeypatch.setattr(backend, "batch_copy", record)
    bundle = frozenset({"layer.0", "layer.1"})
    for prepare in (False, True):
        session = io.begin([plan(target)], writing=False)
        if prepare:
            io.prepare_layer_reads(session, (bundle,))
            assert session.ranges_by_owner is None
        io.copy(session, bundle)
        io.finish(session)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert target.raw == source.raw


def test_range_completion_survives_partial_copy_and_skips_repeated_hooks(monkeypatch):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend, max_segments=1)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    target = ctypes.create_string_buffer(8)
    write = io.begin([plan(source)], writing=True)
    io.copy(write)
    io.finish(write)
    read = io.begin([plan(target)], writing=False)
    native_copy = backend.batch_copy
    calls = []

    def fail_second(gvas, addresses, sizes, direction):
        calls.append(addresses[0])
        if len(calls) == 2:
            return -1
        return native_copy(gvas, addresses, sizes, direction)

    monkeypatch.setattr(backend, "batch_copy", fail_second)
    with pytest.raises(RuntimeError, match="copy failed"):
        io.copy(read, {"layer.0", "layer.1"})
    assert read.completed_ranges == 1
    io.copy(read, {"layer.0"})
    io.copy(read)
    io.finish(read)
    assert calls == [ctypes.addressof(target), ctypes.addressof(target) + 4, ctypes.addressof(target) + 4]
    assert read.completed_ranges == 2
    assert target.raw == source.raw
    assert not backend.leases


def test_incomplete_or_failed_copy_never_published():
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    session = io.begin([plan(source)], writing=True)
    io.copy(session, {"layer.0"})
    with pytest.raises(RuntimeError, match="partially copied"):
        io.finish(session)
    assert not backend.readable and backend.leases == {"shared-key"}
    session = io.begin([plan(source)], writing=True)
    backend.fail_copy = True
    with pytest.raises(RuntimeError, match="copy failed"):
        io.copy(session)
    io.abort(session)
    assert not backend.readable and backend.leases == {"shared-key"}


def test_missing_read_fails_closed():
    backend = MemoryBackend()
    target = ctypes.create_string_buffer(8)
    with pytest.raises(RuntimeError, match="read lease failed"):
        MemcacheStoreLayoutIO(backend).begin([plan(target)], writing=False)
    assert not backend.leases


@pytest.mark.parametrize("existing_reader", [False, True])
def test_mixed_read_lease_failure_releases_only_acquired_leases(monkeypatch, existing_reader):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    buffer = ctypes.create_string_buffer(b"abcdefgh", 8)
    present = plan(buffer)
    missing = replace(present, key="missing")
    writer = io.begin([present], writing=True)
    io.copy(writer)
    io.finish(writer)
    reader = io.begin([present], writing=False) if existing_reader else None
    release = backend.batch_remove_lease

    def strict_release(keys):
        assert set(keys) <= backend.leases, "SDK rejects removing a lease that was not acquired"
        return release(keys)

    monkeypatch.setattr(backend, "batch_remove_lease", strict_release)
    with pytest.raises(RuntimeError, match="read lease failed") as failure:
        io.begin([present, missing], writing=False)
    assert not getattr(failure.value, "__notes__", [])
    assert backend.leases == ({present.key} if existing_reader else set())
    if reader is not None:
        io.abort(reader)
    # A miss must not leave a permanent pending cleanup entry: the same key
    # can subsequently be published and read without restarting the client.
    writer = io.begin([missing], writing=True)
    io.copy(writer)
    io.finish(writer)
    reader = io.begin([missing], writing=False)
    io.copy(reader)
    io.finish(reader)
    assert not backend.leases


def test_two_readers_retain_one_lease_until_last_consumer_finishes():
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    writer = io.begin([plan(source)], writing=True)
    io.copy(writer)
    io.finish(writer)
    targets = [ctypes.create_string_buffer(8) for _ in range(2)]
    first, second = [io.begin([plan(target)], writing=False) for target in targets]
    assert first.deadline == second.deadline
    io.copy(first)
    io.finish(first)
    assert backend.leases == {"shared-key"}
    io.copy(second)
    io.finish(second)
    assert all(target.raw == source.raw for target in targets)
    assert not backend.leases


@pytest.mark.parametrize("writing", [False, True])
def test_cleanup_status_failure_does_not_fail_completed_transfer(monkeypatch, writing):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    writer = io.begin([plan(source)], writing=True)
    io.copy(writer)
    if writing:
        session = writer
    else:
        io.finish(writer)
        target = ctypes.create_string_buffer(8)
        session = io.begin([plan(target)], writing=False)
        io.copy(session)
    calls = []

    def failed_release(keys):
        calls.append(tuple(keys))
        return -3106

    monkeypatch.setattr(backend, "batch_remove_lease", failed_release)
    io.finish(session)
    assert session.closed and session.released
    assert backend.readable == {"shared-key"}
    assert not io._references and not io._pending_release and not io._deadlines
    if not writing:
        assert target.raw == source.raw
    # The failed old cleanup must never release a newly acquired lease.
    reader = io.begin([plan(source)], writing=False)
    io.abort(session)
    io.retry_cleanup()
    assert len(calls) == (0 if writing else 1)
    io.copy(reader)
    io.finish(reader)
    assert len(calls) == (1 if writing else 2)


def test_cleanup_status_failure_preserves_publication_error(monkeypatch):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    session = io.begin([plan(source)], writing=True)
    io.copy(session)
    monkeypatch.setattr(backend, "batch_write_finish", lambda *args: [-1])
    monkeypatch.setattr(backend, "batch_remove_lease", lambda keys: -3106)
    with pytest.raises(RuntimeError, match="publication failed"):
        io.finish(session)
    assert session.closed and not backend.readable
    assert not io._pending_release


def test_partial_allocation_exception_leaves_write_lease_to_ttl(monkeypatch):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend)
    alloc = backend.batch_alloc

    def fail_after_alloc(*args):
        alloc(*args)
        raise RuntimeError("partial allocation")

    monkeypatch.setattr(backend, "batch_alloc", fail_after_alloc)
    source = ctypes.create_string_buffer(b"abcdefgh", 8)
    with pytest.raises(RuntimeError, match="partial allocation"):
        io.begin([plan(source)], writing=True)
    assert backend.leases == {"shared-key"} and not backend.readable


def test_incomplete_readability_response_is_not_a_hit(monkeypatch):
    backend = MemoryBackend()
    monkeypatch.setattr(backend, "batch_is_readable", lambda keys: [])
    with pytest.raises(RuntimeError, match="incomplete readability"):
        MemcacheStoreLayoutIO(backend).batch_is_exist(["missing"])


@pytest.mark.parametrize("fail_second_copy", [False, True])
def test_send_layer_batches_sessions_before_publication(fail_second_copy):
    backend = MemoryBackend()
    io = MemcacheStoreLayoutIO(backend, max_segments=2)
    sources = [ctypes.create_string_buffer(value, 8) for value in (b"abcdefgh", b"ABCDEFGH")]
    sessions = [io.begin([replace(plan(source), key=f"key-{i}")], writing=True) for i, source in enumerate(sources)]
    batches = []

    def send(gvas, addresses, sizes):
        batches.append(sizes.tolist())
        if len(batches) == 2 and fail_second_copy:
            return -1
        return backend.batch_copy(gvas.tolist(), addresses.tolist(), sizes.tolist(), 0)

    io.copy_save_layer(sessions, {"layer.0"}, send)
    assert not backend.readable
    if fail_second_copy:
        with pytest.raises(RuntimeError, match="copy failed"):
            io.copy_save_layer(sessions, {"layer.1"}, send)
        assert [session.completed_ranges for session in sessions] == [1, 1]
        for session in sessions:
            with pytest.raises(RuntimeError, match="partially copied"):
                io.finish(session)
        assert not backend.readable
    else:
        io.copy_save_layer(sessions, {"layer.1"}, send)
        assert not backend.readable
        for i, session in enumerate(sessions):
            io.finish(session)
            assert backend.objects[f"key-{i}"].raw == sources[i].raw
    assert batches == [[4, 4], [4, 4]]
