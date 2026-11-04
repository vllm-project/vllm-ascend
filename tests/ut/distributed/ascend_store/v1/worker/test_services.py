"""Independent Lookup binding and operation-owned drain guarantees."""

import threading
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.align_state_fixtures import make_align_state_topology
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    make_block_hash_mappings,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StoreCommandBatch,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.outcomes import DrainResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.service.lookup import LookupService


def test_lookup_binds_and_queries_without_local_buffers_or_resource_owner():
    topology = make_topology()
    backend = FakeBackend()
    backend.presence = [True, False]
    lookup = LookupService(
        topology, backend, make_coordinator(topology), block_hash_mappings=make_block_hash_mappings(topology)
    )
    assert (
        lookup(
            LookupRequest(end_token=8, transfer_group_ids=(0,), block_hashes=(b"a", b"b"), local_available_tokens=0)
        ).available_end_token
        == 4
    )
    assert [call[0] for call in backend.calls] == ["exists"]
    assert not backend.closed


def test_lookup_resolves_partial_checkpoints_without_local_block_assignments(monkeypatch):
    topology = make_align_state_topology(attention_block_size=16, state_block_size=32, hash_block_size=8, tp_size=1)
    backend = FakeBackend()
    monkeypatch.setattr(backend, "exists", lambda keys: tuple(not key.endswith("@04") for key in keys))
    lookup = LookupService(
        topology, backend, make_coordinator(topology), block_hash_mappings=make_block_hash_mappings(topology)
    )
    result = lookup(
        LookupRequest(
            end_token=32,
            transfer_group_ids=(0, 1),
            block_hashes=(b"\x01", b"\x02", b"\x03", b"\x04"),
            local_available_tokens=0,
        )
    )
    assert result.available_end_token == 24
    assert result.tail_key_boundaries == (TailKeyBoundary(0, 24), TailKeyBoundary(1, 24))


def test_lookup_distinguishes_query_failure_from_malformed_results(monkeypatch):
    topology = make_topology()
    backend = FakeBackend()
    lookup = LookupService(
        topology, backend, make_coordinator(topology), block_hash_mappings=make_block_hash_mappings(topology)
    )
    request = LookupRequest(end_token=4, transfer_group_ids=(0,), block_hashes=(b"a",), local_available_tokens=0)
    monkeypatch.setattr(backend, "exists", Mock(side_effect=RuntimeError("query unavailable")))
    assert lookup(request).available_end_token == 0
    monkeypatch.setattr(backend, "exists", lambda _keys: ())
    with pytest.raises(ValueError, match="result count"):
        lookup(request)


def test_load_drain_preserves_uncollected_notifications_and_stops_submissions():
    worker, resources, _ = make_worker()
    command = LoadCommand("load", 0, 4, ((1,),), (b"a",))
    try:
        worker.begin_step(KVTransferStep((command,)))
        worker.start_load()
        assert worker.load.drain() == DrainResult(True)
        with pytest.raises(RuntimeError, match="no longer accepts"):
            worker.load.submit(())
        assert not resources.closed
        worker.close()
        assert worker.collect_load_result().completed_request_ids == {"load"}
        assert not worker.collect_load_result().completed_request_ids
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_store_drain_observes_later_jobs_after_an_earlier_unknown_source(monkeypatch):
    worker, resources, backend = make_worker()

    def store(keys, addresses, sizes):
        backend.calls.append(("copy", tuple(keys)))
        if keys[0].endswith("@61"):
            raise RuntimeError("source completion unknown")
        return (0,) * len(keys)

    monkeypatch.setattr(backend, "store", store)
    event = FakeEvent()
    event.record()
    worker.store.submit(
        (RangeStoreCommand("unknown", 0, 4, ((1,),), (b"a",), 4, 17),),
    )
    worker.transfer.read_all_layers(lambda: event)
    worker.store.submit(
        (RangeStoreCommand("safe", 0, 4, ((2,),), (b"b",), 4, 18),),
    )
    worker.transfer.read_all_layers(lambda: event)
    drained = worker.store.drain()
    assert not drained.safe_to_release and isinstance(drained.error, RuntimeError)
    assert len([call for call in backend.calls if call[0] == "copy"]) == 2
    assert worker.take_released_store_job_ids() == {18}
    assert not worker.store.drain().safe_to_release
    assert not worker.take_released_store_job_ids()
    with pytest.raises(RuntimeError, match="source release is unknown"):
        worker.close()
    assert resources.executors_stopped and not resources.closed


def test_repeated_store_drain_does_not_repeat_safe_peer_release():
    worker, resources, backend = make_worker(requires_exists=True)
    backend.presence = (0, 1)
    backend.put_result = RuntimeError("source completion unknown")
    commands = (
        RangeStoreCommand("unknown", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),
        RangeStoreCommand("duplicate", 0, 8, ((3, 4),), (b"a", b"b"), 8, 18),
        RangeStoreCommand("present", 0, 4, ((5,),), (b"b",), 4, 19),
    )
    worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
    worker.finish_step()
    assert not worker.store.drain().safe_to_release
    assert worker.take_released_store_job_ids() == {18, 19}
    assert not worker.store.drain().safe_to_release
    assert not worker.take_released_store_job_ids()
    with pytest.raises(RuntimeError, match="source release is unknown"):
        worker.close()
    assert not worker.take_released_store_job_ids()
    assert not resources.closed


@pytest.mark.parametrize("mode", ("error_report", "exception", "unknown"))
def test_worker_uses_drain_reports_and_attempts_both_services(mode, monkeypatch):
    worker, resources, _ = make_worker()
    trace = []
    error = RuntimeError("Store drain failed")

    def drain_store():
        trace.append("store")
        if mode == "exception":
            raise error
        return DrainResult(False, None if mode == "unknown" else error)

    monkeypatch.setattr(worker.store, "drain", drain_store)

    def drain_load():
        trace.append("load")
        return DrainResult(True)

    monkeypatch.setattr(worker.load, "drain", drain_load)
    with pytest.raises(RuntimeError) as raised:
        worker.close()
    if mode == "unknown":
        assert "did not confirm safe memory release" in str(raised.value)
    else:
        assert raised.value is error
    assert trace == ["store", "load"]
    assert resources.executors_stopped and not resources.closed


def test_close_stops_new_work_before_waiting_for_inflight_store(monkeypatch):
    waiting, release, admission_stopped = threading.Event(), threading.Event(), threading.Event()

    class SourceEvent(FakeEvent):
        def synchronize(self):
            waiting.set()
            assert release.wait(timeout=5)
            super().synchronize()

    worker, resources, _ = make_worker(event_factory=SourceEvent)
    original_stop = worker.load.stop_accepting

    def stop_load():
        original_stop()
        admission_stopped.set()

    monkeypatch.setattr(worker.load, "stop_accepting", stop_load)
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    worker.finish_step()
    assert waiting.wait(timeout=2)
    errors = []

    def close():
        try:
            worker.close()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=close)
    thread.start()
    try:
        assert admission_stopped.wait(timeout=2)
        with pytest.raises(RuntimeError, match="closing"):
            worker.lookup(
                LookupRequest(end_token=4, transfer_group_ids=(0,), block_hashes=(b"a",), local_available_tokens=0)
            )
        with pytest.raises(RuntimeError, match="closing"):
            worker.start_load()
        with pytest.raises(RuntimeError, match="no longer accepts"):
            worker.load.submit(())
        with pytest.raises(RuntimeError, match="no longer accepts"):
            worker.store.submit(())
        assert not resources.closed
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive() and not errors
    assert resources.closed and worker.take_released_store_job_ids() == {17}
    worker.end_step()
    worker.close()
