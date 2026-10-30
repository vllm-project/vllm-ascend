"""Empty work, shared routing and native failure safety survive preparation shortcuts."""

from contextlib import suppress
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.bulk_prototype.fixtures import make_worker as make_bulk
from tests.ut.distributed.ascend_store.v1.bulk_prototype.test_gva_writers import cluster, submit_stores
from tests.ut.distributed.ascend_store.v1.bulk_prototype.test_layerwise import layer_names
from tests.ut.distributed.ascend_store.v1.bulk_prototype.test_layerwise import make_worker as make_layerwise
from tests.ut.distributed.ascend_store.v1.helpers import make_topology
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import FakeGVABackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)


@pytest.mark.parametrize("mode", ("bulk", "key_range", "gva"))
@pytest.mark.parametrize("failure", (False, True))
def test_known_empty_or_failed_preparation_needs_no_source_event(mode, failure, monkeypatch):
    record = Mock(side_effect=AssertionError("empty work must not record an Event"))
    worker, _, _ = (
        make_bulk(event_factory=record, requires_exists=True)
        if mode == "bulk"
        else make_layerwise(
            backend=FakeGVABackend() if mode == "gva" else None, event_factory=record, requires_exists=True
        )
    )
    monkeypatch.setattr(worker.transfer.backend, "exists", lambda keys: (True,) * len(keys))
    if failure:
        monkeypatch.setattr(worker.store, "_candidates", Mock(side_effect=RuntimeError("selection failed")))
    try:
        command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        worker.transfer.executor._read.submit(lambda: None).result(timeout=2)
        worker.transfer.read_all_layers(record)
        (result,) = worker.fence_previous_store()
        assert result.source_release_confirmed and not result.keys
        assert result.succeeded is not failure
        assert worker.take_released_store_job_ids() == {17}
        worker.finish_step()
        worker.end_step()
        record.assert_not_called()
    finally:
        worker.close()


@pytest.mark.parametrize("mode", ("bulk", "key_range", "gva"))
def test_empty_load_reports_completion_without_queue_or_backend_work(mode, monkeypatch):
    worker, _, _ = (
        make_bulk() if mode == "bulk" else make_layerwise(backend=FakeGVABackend() if mode == "gva" else None)
    )
    try:
        worker.begin_step(KVTransferStep((LoadCommand("r", 0, 0, ((),), ()),)))
        with monkeypatch.context() as patch:
            submit = Mock(side_effect=AssertionError("empty load must not enqueue work"))
            patch.setattr(worker.transfer.executor._write, "submit", submit)
            worker.start_load()
            assert worker.collect_load_result().completed_request_ids == {"r"}
            submit.assert_not_called()
        worker.end_step()
    finally:
        worker.close()


def test_store_keeps_source_positions_when_exists_filters_objects(monkeypatch):
    worker, _, _ = make_bulk(topology=make_topology(group_ids=(0, 1)), requires_exists=True)
    exists = Mock(side_effect=lambda keys: tuple(key.endswith("@62") for key in keys))
    monkeypatch.setattr(worker.transfer.backend, "exists", exists)
    commands = (
        RangeStoreCommand("first", 0, 8, ((1, 2), (3, 4)), (b"a", b"b"), 8, 17),
        RangeStoreCommand("duplicate", 0, 8, ((5, 6), (7, 8)), (b"a", b"b"), 8, 18),
    )
    try:
        selection = worker.store._prepare(commands)
        exists.assert_called_once()
        assert len(exists.call_args.args[0]) == 4
        assert len(selection.keys) == len(selection.sources) == 2
        assert all(key.endswith("@61") for key in selection.keys)
        assert [request_index for _, _, request_index in selection.sources] == [0, 0]
        assert [block_id for _, block_id, _ in selection.sources] == [1, 3]
    finally:
        worker.close()


def test_gva_fanout_uses_scalar_codes_without_key_range_merges(monkeypatch):
    backend = FakeGVABackend()
    worker, _, _ = make_layerwise(backend=backend)
    commands = (LoadCommand("a", 0, 4, ((1,),), (b"a",)), LoadCommand("b", 0, 4, ((2,),), (b"a",)))
    selection = worker.load._prepare(commands)
    backend.native_store.objects[selection.keys[0]] = (10000, worker.context.remote[0][0]["size_bytes"], True)
    copy = worker.transfer.copy
    results = []

    def observe(*args, **kwargs):
        result = copy(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(worker.transfer, "copy", observe)
    try:
        worker.begin_step(KVTransferStep(commands))
        worker.start_load()
        plan = worker.transfer._write_batch.plan
        assert not plan.key_merges and not plan.regions
        assert not plan.range_indices
        assert all(tuple(plan.result_indices(layer)) == (0, 1) for layer in plan.layers)
        for name in layer_names(worker):
            worker.wait_for_layer_load(name)
        assert results and all(type(result) is int for result in results)
        assert worker.collect_load_result().completed_request_ids == {"a", "b"}
        worker.end_step()
    finally:
        worker.close()


def test_bulk_completion_does_not_build_noop_publication_arguments(monkeypatch):
    worker, _, _ = make_bulk()
    commit = Mock(side_effect=AssertionError("Bulk is already published"))
    revoke = Mock(side_effect=AssertionError("Bulk has no remote lease"))
    monkeypatch.setattr(worker.transfer, "commit_read", commit)
    monkeypatch.setattr(worker.transfer, "revoke_read", revoke)
    try:
        command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        worker.finish_step()
        (result,) = worker.fence_previous_store()
        assert result.succeeded and result.source_release_confirmed
        worker.end_step()
        commit.assert_not_called()
        revoke.assert_not_called()
    finally:
        worker.close()


def test_nonwriters_skip_selection_and_io_but_join_empty_confirmation(monkeypatch):
    with cluster(tp=4, pcp=1, dcp=2, put_step=4) as group:
        nonwriters = [worker for worker in group.workers if all(sl.stop == 0 for sl in worker.store._writers.values())]
        assert len(nonwriters) == 2
        for worker in nonwriters:
            monkeypatch.setattr(worker.store.coordinator, "select_for_store", Mock(side_effect=AssertionError("scan")))
            monkeypatch.setattr(worker.transfer.backend, "exists", Mock(side_effect=AssertionError("exists")))
            monkeypatch.setattr(worker.transfer, "_store_regions", Mock(side_effect=AssertionError("open")))
        submit_stores(group)
        for worker, channel in zip(group.workers, group.channels, strict=True):
            (result,) = worker.fence_previous_store()
            assert result.succeeded and result.source_release_confirmed
            assert worker.take_released_store_job_ids() == {17}
            assert len(channel.calls) == 1
            if worker in nonwriters:
                assert not result.keys and channel.calls == [{}]


@pytest.mark.parametrize("mode", ("key_range", "gva"))
def test_terminal_failure_skips_remaining_reads_without_confirming_source_release(mode, monkeypatch):
    worker, resources, _ = make_layerwise(backend=FakeGVABackend() if mode == "gva" else None)
    copy = worker.transfer.copy
    attempts = []

    def fail_first(plan, layer_id, **kwargs):
        attempts.append(layer_id)
        if len(attempts) == 1:
            kwargs["source_ready"].synchronize()
            kwargs["on_start"]()
            raise RuntimeError("native read lifetime unknown")
        return copy(plan, layer_id, **kwargs)

    monkeypatch.setattr(worker.transfer, "copy", fail_first)
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        for name in layer_names(worker):
            worker.save_layer(name)
        batch = worker.transfer._read_batch
        with pytest.raises(RuntimeError, match="source release is unknown"):
            worker.finish_step()
        assert len(attempts) == 1 and batch.unsafe_objects == {0}
        assert not worker.take_released_store_job_ids()
    finally:
        with suppress(RuntimeError):
            worker.close()
    assert not resources.closed


@pytest.mark.parametrize("mode", ("bulk", "key_range"))
def test_rejected_write_submission_keeps_a_failed_completion_for_drain(mode, monkeypatch):
    worker, resources, _ = make_bulk() if mode == "bulk" else make_layerwise()
    command = LoadCommand("r", 0, 4, ((1,),), (b"a",))
    error = RuntimeError("write queue rejected submission")
    worker.begin_step(KVTransferStep((command,)))
    with monkeypatch.context() as patch:
        patch.setattr(worker.transfer.executor._write, "submit", Mock(side_effect=error))
        with pytest.raises(RuntimeError, match="write queue rejected"):
            worker.start_load()
        (batch,) = worker.load._pending
        assert batch.completion.done() and batch.completion.exception() is error
        assert worker.load.safe_to_release
    with pytest.raises(RuntimeError, match="write queue rejected"):
        worker.close()
    assert resources.closed
