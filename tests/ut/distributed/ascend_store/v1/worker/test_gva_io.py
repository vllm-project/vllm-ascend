"""GVA sessions respect leases, object offsets, and publication visibility."""

from __future__ import annotations

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import (
    FakeGVABackend,
    make_gva_binding,
    make_gva_spec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import (
    GVABackendIO,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import (
    GVAObjectLayout,
    KVPoolResources,
)


def test_gva_initialization_validates_native_capabilities_and_registered_layout(monkeypatch) -> None:
    backend = FakeGVABackend()
    assert backend.native_store is not None
    monkeypatch.setattr(backend.native_store, "batch_write_finish", None)
    monkeypatch.setattr(backend, "batch_write_finish", lambda *args: [0], raising=False)
    with pytest.raises(RuntimeError, match="native batch_write_finish"):
        GVABackendIO(backend, make_gva_spec())

    unavailable = FakeGVABackend()
    unavailable.native_store = None
    with pytest.raises(RuntimeError, match="store is unavailable"):
        GVABackendIO(unavailable, make_gva_spec())

    topology = make_topology(physical_layers=(4, 5))
    registration_backend = FakeBackend()
    resources = KVPoolResources(
        registration_backend,
        make_gva_spec(),
        8,
        topology.groups,
        gva_layout=GVAObjectLayout(
            pipeline_layer_start=4,
            total_hidden_layers=4,
            dcp_rank=1,
            dcp_size=2,
            put_step=2,
        ),
    )
    caches = {layer_name: torch.empty((8, 4), dtype=torch.float32) for layer_name in topology.groups[0].layer_names}
    registration = resources.bind_kv_caches(caches)

    assert registration["base_addresses"] == {0: [cache.data_ptr() for cache in caches.values()]}
    assert registration["block_lengths"] == {0: [16, 16]}
    assert registration["block_strides"] == {0: [16, 16]}
    assert registration["layer_entry_offsets"] == {0: [0, 1, 2]}
    assert registration["object_offsets"] == {0: 2 * 1024 * 1024 + 64}
    assert registration["object_sizes"] == {0: 4 * 1024 * 1024}
    assert registration_backend.calls[-1][0] == "register_buffer"


def test_gva_readability_changes_only_after_publication() -> None:
    backend_io, batch, store = make_gva_binding((1,))
    key = batch.selected_keys()[0]

    assert backend_io.start_store_sessions([key], [64]) == (0,)
    assert not backend_io.exists([key])[0]
    assert backend_io.store_batch(batch, 0)[0].evidence.succeeded
    assert not backend_io.exists([key])[0]
    assert backend_io.commit_store_sessions([key]) == (0,)
    assert backend_io.exists([key])[0]
    assert not backend_io._store_sessions
    assert store.objects[key][2]


def test_gva_load_admission_resolves_after_lease_and_releases_only_owned_keys(monkeypatch) -> None:
    backend_io, batch, store = make_gva_binding()
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
    assert backend_io.start_load_sessions([first, second, "wrong-size"], [64, 64, 64]) == (0, -7, -1)
    completion = backend_io.load_batch(batch.select_keys({first}), 0)[0]
    assert completion.transfer_evidence[0].result_code == 0
    assert next(call for call in store.calls if call[0] == "copy")[1] == (4000,)
    backend_io.finish_load_sessions([first, second, "wrong-size"])
    assert store.calls[-1] == ("release", (first,))
    assert not store.leases


def test_gva_load_copies_each_layer_at_its_leased_object_offset() -> None:
    backend_io, batch, store = make_gva_binding()
    first, second = batch.selected_keys()
    store.objects.update({first: (1000, 64, True), second: (2000, 64, True)})
    assert backend_io.start_load_sessions([first, second], [64, 64]) == (0, 0)
    backend_io.prepare_load_layers(batch)

    for layer_id in (0, 1):
        assert all(item.result_code == 0 for item in backend_io.load_layer(layer_id)[0].transfer_evidence)
    copies = [call for call in store.calls if call[0] == "copy"]
    assert [call[1] for call in copies] == [(1000, 2000), (1032, 2032)]
    assert [call[2] for call in copies] == [(1064, 1192), (2064, 2192)]
    assert [call[3] for call in copies] == [(32, 32), (32, 32)]
    backend_io.finish_load_sessions([first, second])
    assert store.calls[-1] == ("release", (first, second))
    assert not store.leases


def test_gva_load_copy_exception_becomes_unknown_source_evidence() -> None:
    backend_io, batch, store = make_gva_binding(block_ids=(1,))
    key = batch.selected_keys()[0]
    store.objects[key] = (1000, 64, True)

    assert backend_io.start_load_sessions([key], [64]) == (0,)
    store.copy_result = RuntimeError("GVA copy failed")
    completion = backend_io.load_batch(batch, 0)[0]

    assert len(completion.transfer_evidence) == 1
    evidence = completion.transfer_evidence[0]
    assert evidence.result_code is None
    assert (
        evidence.source.group_id,
        evidence.source.block_id,
        evidence.source.physical_layer_ids,
    ) == (0, 1, (0,))
    backend_io.finish_load_sessions([key])
    assert not store.leases


def test_gva_store_admission_owns_only_successful_allocations() -> None:
    backend_io, batch, store = make_gva_binding()
    first, second = batch.selected_keys()
    store.allocation_result = [1000, 0]

    assert backend_io.start_store_sessions([first, second], [64, 64]) == (0, -1)
    assert backend_io._store_sessions == {first: (1000, 64)}
    assert store.calls == [("alloc", (first, second), (64, 64))]
    assert backend_io.revoke_store_sessions([first, second]) == (-1, 0)
    assert not backend_io._store_sessions


def test_gva_copy_normalizes_batch_result_and_unknown_evidence() -> None:
    for native_result, expected_codes, succeeded, released in (
        (0, [0, 0], True, True),
        (-9, [-9, -9], False, True),
        (None, [None, None], False, False),
        (True, [None, None], False, False),
        ([0], [None, None], False, False),
        (RuntimeError("copy failed"), [None, None], False, False),
    ):
        backend_io, batch, store = make_gva_binding()
        keys = list(batch.selected_keys())
        store.copy_result = native_result
        assert backend_io.start_store_sessions(keys, [64, 64]) == (0, 0)
        evidence = backend_io.store_batch(batch, 0)[0].evidence
        assert [item.result_code for item in evidence.transfer_evidence] == expected_codes
        assert all(item.source_release_confirmed is released for item in evidence.transfer_evidence)
        assert evidence.succeeded is succeeded
        assert evidence.source_release_confirmed is released


def test_gva_publication_failure_does_not_hide_copy_source_release() -> None:
    backend_io, batch, store = make_gva_binding((1,))
    store.commit_result = [-8]
    key = batch.selected_keys()[0]

    assert backend_io.start_store_sessions([key], [64]) == (0,)
    assert backend_io.store_batch(batch, 0)[0].evidence.source_release_confirmed
    assert backend_io.commit_store_sessions([key]) == (-8,)
    assert not store.objects[key][2]
    assert backend_io.revoke_store_sessions([key]) == (-1,)
