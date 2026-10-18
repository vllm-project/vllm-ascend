"""GVA sessions respect leases, object offsets, and publication visibility."""

from __future__ import annotations

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import (
    FakeGVABackend,
    make_gva_binding,
    make_gva_spec,
)
from tests.ut.distributed.ascend_store.v1.worker.resource_fixtures import register_caches
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import GVARegion
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import (
    KVPoolResources,
)


def test_gva_initialization_validates_native_capabilities_and_registered_layout(monkeypatch) -> None:
    backend = FakeGVABackend()
    assert backend.native_store is not None
    monkeypatch.setattr(backend.native_store, "batch_write_finish", None)
    monkeypatch.setattr(backend, "batch_write_finish", lambda *args: [0], raising=False)
    with pytest.raises(RuntimeError, match="native batch_write_finish"):
        backend.create_sessions()

    unavailable = FakeGVABackend()
    unavailable.native_store = None
    with pytest.raises(RuntimeError, match="store is unavailable"):
        unavailable.create_sessions()

    topology = make_topology(physical_layers=(4, 5))
    registration_backend = FakeBackend()
    resources = KVPoolResources(
        registration_backend,
        make_gva_spec(),
        8,
        topology.groups,
    )
    caches = {layer_name: torch.empty((8, 4), dtype=torch.float32) for layer_name in topology.groups[0].layer_names}
    registration = register_caches(resources, caches)

    assert registration["base_addresses"] == {0: [cache.data_ptr() for cache in caches.values()]}
    assert registration["block_lengths"] == {0: [16, 16]}
    assert registration["block_strides"] == {0: [16, 16]}
    assert registration["layer_entry_offsets"] == {0: [0, 1, 2]}
    assert "object_offsets" not in registration and "object_sizes" not in registration
    assert registration_backend.calls[-1][0] == "register_buffer"


def test_gva_readability_changes_only_after_publication() -> None:
    worker, batch, store, sessions = make_gva_binding((1,))
    key = batch.selected_keys()[0]

    assert sessions.start_store([key], [64]) == (0,)
    assert not worker._exists([key])[0]
    worker._prepare_store_layers(batch)
    assert worker._copy_store_layer(FakeEvent(), 0).result_codes == 0
    assert not worker._exists([key])[0]
    assert sessions.commit_store([key]) == (0,)
    assert worker._exists([key])[0]
    assert not sessions.pending_store_keys
    assert store.objects[key][2]


def test_gva_load_admission_resolves_after_lease_and_releases_only_owned_keys(monkeypatch) -> None:
    worker, batch, store, sessions = make_gva_binding()
    first, second = batch.selected_keys()
    store.objects.update(
        {
            first: (1000, 64, True),
            second: (2000, 64, True),
            "wrong-size": (3000, 8, True),
        }
    )
    store.lease_result = [0, -7, 0]
    original_add_lease = store.batch_add_lease

    def acquire(keys):
        store.objects[first] = (4000, 64, True)
        return original_add_lease(keys)

    monkeypatch.setattr(store, "batch_add_lease", acquire)
    assert sessions.start_load([first, second, "wrong-size"], [64, 64, 64]) == (0, -7, -1)
    worker._prepare_load_layers(batch.select_keys({first}))
    completion = worker._copy_load_layer(0)[0]
    assert completion.request_id == "request" and completion.failures == ()
    assert next(call for call in store.calls if call[0] == "copy")[1] == (4000,)
    sessions.finish_load([first, second, "wrong-size"])
    assert store.calls[-1] == ("release", (first,))
    assert not store.leases


def test_gva_load_copies_each_layer_at_its_leased_object_offset() -> None:
    worker, batch, store, sessions = make_gva_binding()
    first, second = batch.selected_keys()
    store.objects.update({first: (1000, 64, True), second: (2000, 64, True)})
    assert sessions.start_load([first, second], [64, 64]) == (0, 0)
    worker._prepare_load_layers(batch)

    for layer_id in (0, 1):
        assert worker._copy_load_layer(layer_id)[0].failures == ()
    copies = [call for call in store.calls if call[0] == "copy"]
    assert [call[1] for call in copies] == [(1000, 2000), (1032, 2032)]
    assert [call[2] for call in copies] == [(1064, 1192), (2064, 2192)]
    assert [call[3] for call in copies] == [(32, 32), (32, 32)]
    sessions.finish_load([first, second])
    assert store.calls[-1] == ("release", (first, second))
    assert not store.leases


def test_gva_load_copy_exception_becomes_unknown_source_evidence() -> None:
    worker, batch, store, sessions = make_gva_binding(block_ids=(1,))
    key = batch.selected_keys()[0]
    store.objects[key] = (1000, 64, True)

    assert sessions.start_load([key], [64]) == (0,)
    store.copy_result = RuntimeError("GVA copy failed")
    worker._prepare_load_layers(batch)
    completion = worker._copy_load_layer(0)[0]

    assert len(completion.failures) == 1
    evidence = completion.failures[0]
    assert evidence.result_code is None
    assert (
        evidence.source.group_id,
        evidence.source.block_id,
        evidence.source.physical_layer_ids,
    ) == (0, 1, (0,))
    sessions.finish_load([key])
    assert not store.leases


def test_gva_store_admission_owns_only_successful_allocations() -> None:
    worker, batch, store, sessions = make_gva_binding()
    first, second = batch.selected_keys()
    store.allocation_result = [1000, 0]

    assert sessions.start_store([first, second], [64, 64]) == (0, -1)
    assert sessions.store_regions((first,), 64) == (GVARegion(1000, 64),)
    assert store.calls == [("alloc", (first, second), (64, 64))]
    assert sessions.revoke_store([first, second]) == (-1, 0)
    assert sessions.pending_store_keys == (first,)
    with pytest.raises(RuntimeError, match="unavailable"):
        sessions.store_regions((first,), 64)


def test_gva_copy_normalizes_batch_result_and_unknown_evidence() -> None:
    for native_result, expected_codes, succeeded, released in (
        (0, [0, 0], True, True),
        (-9, [-9, -9], False, True),
        (None, [None, None], False, False),
        (True, [None, None], False, False),
        ([0], [None, None], False, False),
        (RuntimeError("copy failed"), [None, None], False, False),
    ):
        worker, batch, store, sessions = make_gva_binding()
        keys = list(batch.selected_keys())
        store.copy_result = native_result
        assert sessions.start_store(keys, [64, 64]) == (0, 0)
        worker._prepare_store_layers(batch)
        result = worker._copy_store_layer(FakeEvent(), 0)
        assert [result.result_codes] * len(keys) == expected_codes
        assert (result.result_codes == 0 and result.error is None) is succeeded
        assert result.source_release_confirmed is released


def test_gva_publication_failure_does_not_hide_copy_source_release() -> None:
    worker, batch, store, sessions = make_gva_binding((1,))
    store.commit_result = [-8]
    key = batch.selected_keys()[0]

    assert sessions.start_store([key], [64]) == (0,)
    worker._prepare_store_layers(batch)
    assert worker._copy_store_layer(FakeEvent(), 0).source_release_confirmed
    assert sessions.commit_store([key]) == (-8,)
    assert not store.objects[key][2]
    assert sessions.revoke_store([key]) == (-1,)
    assert sessions.pending_store_keys == (key,)
