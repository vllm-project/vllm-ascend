"""Worker admission, independent execution queues, and per-step source ownership."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_topology,
    make_worker,
)
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import make_gva_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import LoadCommand, RangeStoreCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import arguments


@pytest.mark.parametrize("scenario", ("no_commands", "no_candidates", "all_present", "all_starts_failed"))
def test_known_empty_store_never_creates_a_step_session_or_layer_work(scenario, monkeypatch):
    backend = FakeBackend()
    if scenario == "all_starts_failed":
        backend.store_session_start_result = [-7]
    record = Mock(side_effect=FakeEvent)
    gate = Mock(side_effect=FakeLoadStartGate)
    worker, resources, _ = make_worker(
        backend,
        layerwise=True,
        physical_layers=(2, 5, 9),
        requires_exists_before_put=scenario == "all_present",
        source_ready_event_factory=record,
        start_gate_factory=gate,
    )
    io = worker._layerwise_backend_io
    prepare = Mock(side_effect=io.prepare_store_layers)
    monkeypatch.setattr(io, "prepare_store_layers", prepare)
    end_token = 0 if scenario == "no_candidates" else 4
    command = RangeStoreCommand("request", TokenRange(0, end_token), ((1,),), (b"a",), 4, 17)
    commands = () if scenario == "no_commands" else (command,)
    try:
        begin_step(worker, store=commands)
        # The empty result is known before the hooks; this barrier is not part of dispatch.
        worker._store_executor.submit(lambda: None).result(timeout=5)
        assert worker._session is None
        submit = Mock(side_effect=worker._store_executor.submit)
        monkeypatch.setattr(worker._store_executor, "submit", submit)
        for layer in (2, 2, 5, 9):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
            worker.save_layer(f"layers.{layer}.group.0")
        assert worker._session is None
        submit.assert_not_called()
        record.assert_not_called()
        gate.assert_not_called()
        prepare.assert_not_called()
        worker.finish_step()
        worker.fence_previous_store()
        calls = [call[0] for call in backend.calls]
        assert calls.count("batch_put_start") == int(scenario == "all_starts_failed")
        assert "batch_copy_put" not in calls
        assert "batch_commit" not in calls
        assert worker.take_released_store_job_ids() == ({17} if commands else set())
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_all_present_store_exits_before_materialization_or_backend_sessions(use_gva, monkeypatch):
    if use_gva:
        worker, resources, backend = make_gva_worker()
        command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
        candidates = worker._build_store_candidates((command,))
        for group in candidates.groups:
            object_size = worker._bound_layerwise_projection.groups[group.group_id].object_size
            for axis in group.key_axes:
                for key in axis:
                    backend.objects[key] = (10000, object_size, True)
    else:
        worker, resources, backend = make_worker(layerwise=True, requires_exists_before_put=True)
        command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    materialize = Mock(side_effect=AssertionError("All-present keys must exit before materialization"))
    start = Mock(side_effect=AssertionError("Presence checks must not acquire Backend sessions"))
    monkeypatch.setattr(worker, "_materialize_store_candidates", materialize)
    monkeypatch.setattr(worker._layerwise_backend_io, "start_store_sessions", start)
    record = Mock(side_effect=FakeEvent)
    monkeypatch.setattr(worker, "_source_ready_event_factory", record)
    try:
        begin_step(worker, store=(command,))
        worker._store_executor.submit(lambda: None).result(timeout=5)
        for name in worker._layer_ids_by_name:
            worker.wait_for_layer_load(name)
            worker.save_layer(name)
        assert worker._session is None
        record.assert_not_called()
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {17}
        materialize.assert_not_called()
        start.assert_not_called()
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("scenario", ("no_commands", "no_objects", "all_starts_failed"))
def test_known_empty_load_never_creates_a_step_session_or_compute_gates(scenario):
    backend = FakeBackend()
    if scenario == "all_starts_failed":
        backend.session_start_result = [-7]
    gate = Mock(side_effect=FakeLoadStartGate)
    worker, resources, _ = make_worker(backend, layerwise=True, start_gate_factory=gate)
    end_token = 0 if scenario == "no_objects" else 4
    command = LoadCommand("request", TokenRange(0, end_token), ((1,),), (b"a",))
    commands = () if scenario == "no_commands" else (command,)
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        for layer in (0, 0, 1):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
        assert worker._session is None
        gate.assert_not_called()
        calls = [call[0] for call in backend.calls]
        assert calls.count("batch_get_start") == int(scenario == "all_starts_failed")
        assert "batch_copy_get" not in calls
        assert "batch_get_end" not in calls
        worker.collect_load_result()
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("start_codes", ((0, 0), (0, -7), (-6, -7)))
@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_store_builds_shared_layer_information_only_after_backend_admission(start_codes, use_gva, monkeypatch):
    topology = make_topology(physical_layers=(2, 5, 9))
    if use_gva:
        worker, resources, backend = make_gva_worker(topology=topology)
        backend.allocation_result = [10000 * (index + 1) if code == 0 else 0 for index, code in enumerate(start_codes)]
    else:
        backend = FakeBackend()
        backend.store_session_start_result = list(start_codes)
        worker, resources, _ = make_worker(backend, layerwise=True, topology=topology)
    io = worker._layerwise_backend_io
    prepare = Mock(side_effect=io.prepare_store_layers)
    monkeypatch.setattr(io, "prepare_store_layers", prepare)
    make_group = arguments.make_layer_transfer_group

    def make_admitted_group(group):
        assert any(call[0] == ("alloc" if use_gva else "batch_put_start") for call in backend.calls)
        assert len(group.selected_keys()) == start_codes.count(0)
        return make_group(group)

    shared_group = Mock(side_effect=make_admitted_group)
    monkeypatch.setattr(arguments, "make_layer_transfer_group", shared_group)
    plans = []
    copy = io.store_layer

    def copy_with_shared_plan(event, layer):
        plans.append(io._store_plan)
        return copy(event, layer)

    monkeypatch.setattr(io, "store_layer", copy_with_shared_plan)
    commands = (
        RangeStoreCommand("first", TokenRange(0, 4), ((1,),), (b"a",), 4, 17),
        RangeStoreCommand("second", TokenRange(0, 4), ((2,),), (b"b",), 4, 18),
    )
    try:
        begin_step(worker, store=commands)
        for layer in (2, 5, 9):
            worker.save_layer(f"layers.{layer}.group.0")
        worker.finish_step()
        assert shared_group.call_count == int(0 in start_codes)
        assert prepare.call_count == int(0 in start_codes)
        assert len(plans) == (3 if 0 in start_codes else 0)
        assert all(plan is plans[0] for plan in plans)
        copies = [call for call in backend.calls if call[0] == ("copy" if use_gva else "batch_copy_put")]
        assert all(len(call[1]) == start_codes.count(0) for call in copies)
        assert worker.take_released_store_job_ids() == {17, 18}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_store_admission_does_not_block_model_hooks_and_keeps_recorded_source_boundary(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True, physical_layers=(0,))
    admitted = threading.Event()
    continue_admission = threading.Event()
    original = worker._admitted_store_keys
    records = []

    def admission(keys):
        admitted.set()
        assert continue_admission.wait(timeout=5)
        return original(keys)

    def event():
        recorded = FakeEvent()
        records.append(recorded)
        return recorded

    monkeypatch.setattr(worker, "_admitted_store_keys", admission)
    monkeypatch.setattr(worker, "_source_ready_event_factory", event)
    try:
        command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
        begin_step(worker, store=(command,))
        assert admitted.wait(timeout=2)
        worker.save_layer("layers.0.group.0")
        assert len(records) == 1 and records[0].recorded and not records[0].synchronized
        assert not any(call[0] == "batch_put_start" for call in backend.calls)
        continue_admission.set()
        worker.finish_step()
        assert records[0].synchronized
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        continue_admission.set()
        worker.close()
    assert resources.closed


def test_blocked_store_does_not_block_load_and_retains_state_after_end_step(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True, start_gate_factory=FakeLoadStartGate)
    store_started = threading.Event()
    release_store = threading.Event()
    copy = worker._layerwise_backend_io.store_layer

    def blocked_copy(event, layer):
        if layer == 0:
            store_started.set()
            assert release_store.wait(timeout=5)
        return copy(event, layer)

    monkeypatch.setattr(worker._layerwise_backend_io, "store_layer", blocked_copy)
    try:
        begin_step(
            worker,
            load=(LoadCommand("load", TokenRange(0, 4), ((2,),), (b"b",)),),
            store=(RangeStoreCommand("store", TokenRange(0, 4), ((1,),), (b"a",), 4, 17),),
        )
        worker.start_load()
        worker.wait_for_layer_load("layers.0.group.0")
        worker.save_layer("layers.0.group.0")
        assert store_started.wait(timeout=2)
        worker._session._get_or_create_compute_start_gate(0).open()
        worker.wait_for_layer_load("layers.1.group.0")
        worker.save_layer("layers.1.group.0")
        assert sum(call[0] == "batch_copy_get" for call in backend.calls) == 2
        assert not any(call[0] == "batch_copy_put" for call in backend.calls)
        worker.end_step()
        assert worker._session is None and worker._store_state is None
        assert worker.take_released_store_job_ids() == set()
        release_store.set()
        (completion,) = worker.fence_previous_store()
        assert completion.evidence.succeeded and completion.evidence.source_release_confirmed
        assert completion.evidence.transfer_evidence[0].source.physical_layer_ids == (0, 1)
        assert worker.take_released_store_job_ids() == {17}
    finally:
        release_store.set()
        worker.close()
    assert resources.closed


def test_escaped_store_task_error_revokes_objects_without_claiming_source_release(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True, physical_layers=(0,))
    error = RuntimeError("copy escaped before source completion was confirmed")

    def escaped_copy(event, layer):
        raise error

    monkeypatch.setattr(worker._layerwise_backend_io, "store_layer", escaped_copy)
    begin_step(worker, store=(RangeStoreCommand("store", TokenRange(0, 4), ((1,),), (b"a",), 4, 17),))
    worker.save_layer("layers.0.group.0")
    worker._finalize_layerwise_store(worker._active_step)
    (completion,) = worker._pending_store_batch.future.result(timeout=5)
    assert completion.evidence.error is error
    assert not completion.evidence.succeeded and not completion.evidence.source_release_confirmed
    assert any(call[0] == "batch_revoke" for call in backend.calls)
    assert not any(call[0] == "batch_commit" for call in backend.calls)
    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.fence_previous_store()
    assert worker.take_released_store_job_ids() == set()
    with pytest.raises(RuntimeError, match="cannot continue after a fatal Store failure"):
        worker.close()
    assert not resources.closed
