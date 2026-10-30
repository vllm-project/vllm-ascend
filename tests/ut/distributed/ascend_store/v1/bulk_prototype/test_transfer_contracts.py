"""Consumption barriers, uniform readiness and read dependencies across empty steps."""

import threading
from contextlib import suppress

import pytest

from tests.ut.distributed.ascend_store.v1.bulk_prototype.fixtures import make_worker as make_bulk
from tests.ut.distributed.ascend_store.v1.bulk_prototype.test_layerwise import make_worker as make_layerwise
from tests.ut.distributed.ascend_store.v1.helpers import FakeEvent, make_topology
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import FakeGVABackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)


@pytest.mark.parametrize("async_load", (False, True))
def test_bulk_waits_before_consumption_only_in_synchronous_mode(async_load):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    worker, resources, backend = make_bulk(async_load=async_load)
    errors = []

    def blocked_get(keys, addresses, sizes):
        entered.set()
        assert release.wait(timeout=5)
        return [0] * len(keys)

    backend.get = blocked_get
    worker.begin_step(KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),)))

    def start():
        try:
            worker.start_load()
        except BaseException as error:
            errors.append(error)
        finally:
            returned.set()

    thread = threading.Thread(target=start)
    try:
        thread.start()
        assert entered.wait(timeout=2)
        if async_load:
            assert returned.wait(timeout=2)
        else:
            assert not returned.is_set()
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive() and not errors
        worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
        assert worker.collect_load_result().completed_request_ids == {"r"}
        worker.end_step()
    finally:
        release.set()
        thread.join(timeout=2)
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("failure_phase", ("open", "layer"))
def test_grouped_layerwise_failure_stops_the_first_consumer_and_releases_leases(failure_phase):
    topology = make_topology(group_ids=(0, 1), physical_layers=(2, 5))
    worker, resources, backend = make_layerwise(topology=topology)
    if failure_phase == "open":
        backend.session_start_result = [0, -7]
    else:
        backend.session_copy_result = [0, -7]
    worker.begin_step(KVTransferStep((LoadCommand("r", 0, 4, ((1,), (2,)), (b"a",)),)))
    try:
        with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
            worker.start_load()
            worker.wait_for_layer_load("layers.2.group.0")
        if failure_phase == "open":
            assert not any(call[0] == "batch_copy_get" for call in backend.calls)
    finally:
        with suppress(RuntimeError):
            worker.end_step()
        with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
            worker.close()
    assert resources.closed
    assert len([call for call in backend.calls if call[0] == "batch_get_end"]) == 1


@pytest.mark.parametrize("async_load", (False, True))
def test_grouped_bulk_failure_is_fatal_at_sync_consumption_but_reported_to_async_scheduler(async_load):
    worker, resources, backend = make_bulk(topology=make_topology(group_ids=(0, 1)), async_load=async_load)
    backend.get_result = [0, -7]
    worker.begin_step(KVTransferStep((LoadCommand("r", 0, 4, ((1,), (2,)), (b"a",)),)))
    try:
        if async_load:
            worker.start_load()
            worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
            result = worker.collect_load_result()
            assert result.completed_request_ids == result.failed_request_ids == {"r"}
        else:
            with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
                worker.start_load()
        worker.end_step()
    finally:
        if async_load:
            worker.close()
        else:
            with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
                worker.close()
    assert resources.closed


@pytest.mark.parametrize("mode", ("bulk", "key_range", "gva"))
def test_services_submit_without_events_and_transfer_consumes_one_all_ready_boundary(mode):
    worker, resources, _ = (
        make_bulk() if mode == "bulk" else make_layerwise(backend=FakeGVABackend() if mode == "gva" else None)
    )
    caller = threading.get_ident()
    event = FakeEvent()
    recordings = []

    def record():
        assert threading.get_ident() == caller
        recordings.append(caller)
        event.record()
        return event

    worker.transfer.begin_step()
    try:
        worker.store.submit((RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17),))
        worker.transfer.read_all_layers(record)
        worker.transfer.read_all_layers(record)
        (result,) = worker.store.fence()
        assert result.succeeded and result.source_release_confirmed
        assert event.synchronized and recordings == [caller]
        assert worker.take_released_store_job_ids() == {17}
    finally:
        worker.close()
    assert resources.closed


def test_empty_steps_preserve_the_previous_read_without_waiting_until_another_store(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockedReady(FakeEvent):
        def synchronize(self):
            entered.set()
            assert release.wait(timeout=5)
            super().synchronize()

    worker, resources, _ = make_layerwise(topology=make_topology(physical_layers=(2,)), event_factory=BlockedReady)
    command = RangeStoreCommand("first", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    worker.save_layer("layers.2.group.0")
    assert entered.wait(timeout=2)
    worker.end_step()
    thread = None
    try:
        for _ in range(2):
            returned = threading.Event()

            def begin_empty(done=returned):
                worker.begin_step(KVTransferStep())
                done.set()

            thread = threading.Thread(target=begin_empty)
            thread.start()
            assert returned.wait(timeout=2)
            thread.join(timeout=2)
            worker.end_step()
        assert worker.store.pending and not worker.take_released_store_job_ids()
        fenced, returned = threading.Event(), threading.Event()
        fence = worker.store.fence

        def observe_fence(*args):
            fenced.set()
            return fence(*args)

        monkeypatch.setattr(worker.store, "fence", observe_fence)
        next_command = RangeStoreCommand("next", 0, 4, ((2,),), (b"b",), 4, 18)

        def begin_next_store():
            worker.begin_step(KVTransferStep(store=StoreCommandBatch((next_command,))))
            returned.set()

        thread = threading.Thread(target=begin_next_store)
        thread.start()
        assert fenced.wait(timeout=2) and not returned.is_set()
        release.set()
        assert returned.wait(timeout=2)
        thread.join(timeout=2)
        assert worker.take_released_store_job_ids() == {17}
        worker.save_layer("layers.2.group.0")
        worker.finish_step()
        worker.end_step()
        assert worker.take_released_store_job_ids() == {18}
    finally:
        release.set()
        if thread is not None:
            thread.join(timeout=2)
        worker.close()
    assert resources.closed
