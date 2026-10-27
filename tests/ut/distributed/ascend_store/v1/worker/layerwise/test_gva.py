"""Tests for worker.layerwise.gva."""

from __future__ import annotations

import logging
from contextlib import nullcontext

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    begin_step,
    make_reachability,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.infrastructure.resources_fixtures import register_caches
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import (
    FakeGVABackend,
    make_gva_binding,
    make_gva_spec,
    make_gva_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import GVARegion
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    LookupRequest,
    RangeStoreCommand,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import (
    KVPoolResources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
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
    resources = KVPoolResources(registration_backend, make_gva_spec(), 8, topology.groups)
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
    store.objects.update({first: (1000, 64, True), second: (2000, 64, True), "wrong-size": (3000, 8, True)})
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
    worker, batch, store, sessions = make_gva_binding(local_block_ids=(1,))
    key = batch.selected_keys()[0]
    store.objects[key] = (1000, 64, True)

    assert sessions.start_load([key], [64]) == (0,)
    store.copy_result = RuntimeError("GVA copy failed")
    worker._prepare_load_layers(batch)
    completion = worker._copy_load_layer(0)[0]

    assert len(completion.failures) == 1
    evidence = completion.failures[0]
    assert evidence.result_code is None
    assert (evidence.source.group_id, evidence.source.local_block_id, evidence.source.physical_layer_ids) == (
        0,
        1,
        (0,),
    )
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


def test_gva_worker_close_unregisters_exact_region_before_backend_close() -> None:
    backend = FakeGVABackend()
    topology = make_topology(physical_layers=(0,))
    backend_spec = make_gva_spec()
    resources = KVPoolResources(backend, backend_spec, 8, topology.groups)
    worker = GVALayerwiseWorker(
        topology,
        resources,
        reachability=make_reachability(topology),
        source_ready_event_factory=FakeEvent,
    )
    cache = torch.empty((8, 4), dtype=torch.float32)

    worker.bind_kv_caches({topology.groups[0].layer_names[0]: cache})
    worker.close()

    lifecycle = [call for call in backend.calls if call[0] in ("register_buffer", "unregister_buffer", "backend_close")]
    assert lifecycle == [
        ("register_buffer", (cache.data_ptr(),), (cache.nbytes,)),
        ("unregister_buffer", cache.data_ptr(), cache.nbytes),
        ("backend_close",),
    ]


def test_gva_worker_publishes_after_all_layers_and_reports_job_release() -> None:
    worker, resources, store = make_gva_worker()
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, store_job_id=17)
    begin_step(worker, store=(command,))

    worker.save_layer("layers.0.group.0")
    assert worker._store_executor is not None
    worker._store_executor.submit(lambda: None).result()
    assert [call[0] for call in store.calls].count("copy") == 1
    assert not any(call[0] == "publish" for call in store.calls)

    worker.save_layer("layers.1.group.0")
    worker._store_executor.submit(lambda: None).result()
    worker.finish_step()
    assert [call[0] for call in store.calls if call[0] in ("alloc", "copy", "publish")] == [
        "alloc",
        "copy",
        "copy",
        "publish",
    ]
    assert all(region[2] for region in store.objects.values())
    assert worker.take_released_store_job_ids() == {17}
    worker.close()
    assert resources.closed


@pytest.mark.parametrize("failure", ("allocation", "copy", "publication"))
def test_gva_store_failure_is_a_cache_miss_and_next_step_can_publish(failure: str, monkeypatch, caplog) -> None:
    monkeypatch.setattr(worker_module, "logger", logging.getLogger(__name__))
    worker, resources, store = make_gva_worker()
    if failure == "allocation":
        store.allocation_result = [0]
    elif failure == "copy":
        store.copy_result = -9
    else:
        store.commit_result = [-8]
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, store_job_id=17)
    begin_step(worker, store=(command,))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()

    assert "KV cache Store failed for request request (job 17)" in caplog.text
    if failure == "allocation":
        assert "Store session start failed" in caplog.text
        assert "result code -1" in caplog.text
    assert worker.take_released_store_job_ids() == {17}
    if failure == "copy":
        assert not any(call[0] == "publish" for call in store.calls)
    assert not any(region[2] for region in store.objects.values())
    assert worker.lookup(LookupRequest(0, 4, (0,), (b"a",))).available_end_token == 0
    worker.end_step()

    store.allocation_result = None
    store.copy_result = 0
    store.commit_result = None
    next_command = RangeStoreCommand("next", 0, 4, ((3,),), (b"b",), 4, store_job_id=18)
    begin_step(worker, store=(next_command,))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()
    assert worker.take_released_store_job_ids() == {18}
    assert worker.lookup(LookupRequest(0, 4, (0,), (b"b",))).available_end_token == 4
    cleanup_unconfirmed = failure != "allocation"
    with pytest.raises(RuntimeError, match="cleanup is unconfirmed") if cleanup_unconfirmed else nullcontext():
        worker.close()
    assert resources.closed is not cleanup_unconfirmed
    assert bool(worker._backend_sessions.pending_store_keys) == cleanup_unconfirmed


def test_gva_unconfirmed_revoke_survives_steps_without_reopening_or_reusing_the_address() -> None:
    worker, resources, store = make_gva_worker()
    store.copy_result = -9
    try:
        for job_id in (17, 18):
            command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, job_id)
            begin_step(worker, store=(command,))
            worker.save_layer("layers.0.group.0")
            worker.save_layer("layers.1.group.0")
            worker.finish_step()
            assert worker.take_released_store_job_ids() == {job_id}
            assert len(worker._backend_sessions.pending_store_keys) == 1
            assert worker._store_plan is None
            worker.end_step()
        # The second attempt cannot acquire or copy through the abandoned session.
        assert sum(item[0] == "alloc" for item in store.calls) == 1
        assert sum(item[0] == "copy" for item in store.calls) == 2
        assert not any(item[0] == "publish" for item in store.calls)
    finally:
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
            worker.close()
    assert not resources.closed
    assert worker._load_executor.stopped and worker._store_executor.stopped
