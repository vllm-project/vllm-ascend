"""Framework finalization and failed hooks must retain tasks without retaining active steps."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import vllm.v1.worker.kv_connector_model_runner_mixin as runner

from tests.ut.distributed.ascend_store.v1.helpers import FakeLoadStartGate, make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import AscendStoreV1Connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)


def connect(worker, monkeypatch, *, layerwise=False, async_load=False):
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.worker = worker
    connector.scheduler = connector.lookup_server = None
    connector._use_layerwise = layerwise
    connector._load_async = async_load and not layerwise
    connector._pending_load_result = connector._connector_metadata = None
    connector._released_store_job_ids = set()
    monkeypatch.setattr(runner, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(runner, "get_forward_context", lambda: None)
    return connector


def output(step=None, *, sync=False):
    return SimpleNamespace(
        kv_connector_metadata=step or KVTransferStep(), has_sync_kv_loads=sync, finished_req_ids=set()
    )


@pytest.mark.parametrize("layerwise", (False, True))
def test_real_no_forward_finalization_skips_wait_for_save(monkeypatch, layerwise):
    record = Mock(side_effect=AssertionError("An empty step has no source to record"))
    worker, resources, _ = make_worker(layerwise=layerwise, source_ready_event_factory=record)
    connector = connect(worker, monkeypatch, layerwise=layerwise)
    monkeypatch.setattr(connector, "wait_for_save", Mock(side_effect=AssertionError("No-forward skips this hook")))
    try:
        connector.handle_preemptions(KVTransferStep())
        with runner.KVConnectorModelRunnerMixin._get_kv_connector_output(output(), wait_for_save=False) as result:
            pass
        assert result.finished_recving == set()
        assert result.invalid_block_ids == set()
        assert connector._connector_metadata is None and worker._step is None
        if layerwise:
            assert not worker._executor.active
        record.assert_not_called()
    finally:
        connector.shutdown()
    assert resources.closed


def test_async_load_finishes_after_clear_and_is_collected_by_next_empty_step(monkeypatch):
    worker, resources, backend = make_worker(async_load=True)
    connector = connect(worker, monkeypatch, async_load=True)
    entered, release = threading.Event(), threading.Event()
    load = backend.load

    def blocked_load(*args):
        entered.set()
        assert release.wait(5)
        return load(*args)

    monkeypatch.setattr(backend, "load", blocked_load)
    step = KVTransferStep((LoadCommand("late", 0, 4, ((1,),), (b"a",)),))
    try:
        with runner.KVConnectorModelRunnerMixin._get_kv_connector_output(output(step), wait_for_save=False) as result:
            pass
        assert entered.wait(2)
        assert result.finished_recving == set()
        assert worker._step is None and worker.load._pending
        completion = worker.load._pending[0].completion
        assert not completion.done()
        release.set()
        completion.result(timeout=2)
        for expected in ({"late"}, set()):
            connector.handle_preemptions(KVTransferStep())
            with runner.KVConnectorModelRunnerMixin._get_kv_connector_output(output(), wait_for_save=False) as result:
                pass
            assert result.finished_recving == expected
            assert not result.invalid_block_ids
        assert not worker.load._pending
    finally:
        release.set()
        connector.shutdown()
    assert resources.closed


def test_layerwise_begin_failure_rolls_back_execution_and_connector_metadata(monkeypatch):
    worker, resources, _ = make_worker(layerwise=True)
    connector = connect(worker, monkeypatch, layerwise=True)
    error = RuntimeError("read queue rejected submission")
    monkeypatch.setattr(worker._executor._read, "submit", Mock(side_effect=error))
    step = KVTransferStep(store=StoreCommandBatch((RangeStoreCommand("s", 0, 4, ((1,),), (b"a",), 4),)))
    with pytest.raises(RuntimeError) as raised:
        connector.bind_connector_metadata(step)
    assert raised.value is error
    assert worker._step is None and not worker._executor.active
    assert connector._connector_metadata is None
    connector.clear_connector_metadata()
    with pytest.raises(RuntimeError) as raised:
        connector.shutdown()
    assert raised.value is error and resources.closed


def test_clear_detaches_all_step_state_even_when_load_is_incomplete(monkeypatch):
    worker, resources, _ = make_worker(layerwise=True)
    connector = connect(worker, monkeypatch, layerwise=True)
    connector.bind_connector_metadata(KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),)))
    connector.start_load_kv(None)
    with pytest.raises(RuntimeError, match="did not reach physical layers") as raised:
        connector.clear_connector_metadata()
    assert connector._connector_metadata is None and worker._step is None
    assert not worker._executor.active
    assert worker.transfer._write_batch is None
    connector.clear_connector_metadata()
    with pytest.raises(RuntimeError) as closed:
        connector.shutdown()
    assert closed.value is raised.value and resources.closed


@pytest.mark.parametrize("sync", (False, True))
@pytest.mark.parametrize("defer_finalize", (False, True))
def test_forward_exception_remains_primary_and_aborts_before_later_hooks(monkeypatch, sync, defer_finalize):
    worker, resources, _ = make_worker(layerwise=True, start_gate_factory=FakeLoadStartGate)
    connector = connect(worker, monkeypatch, layerwise=True)
    step = KVTransferStep(
        (LoadCommand("r", 0, 4, ((1,),), (b"a",)),) if sync else (),
        StoreCommandBatch((RangeStoreCommand("s", 0, 4, ((2,),), (b"b",), 4, 17),)),
    )
    error = ValueError("forward failed")
    with (
        pytest.raises(ValueError) as raised,
        runner.KVConnectorModelRunnerMixin._get_kv_connector_output(
            output(step, sync=sync), defer_finalize=defer_finalize
        ),
    ):
        if sync:
            connector.wait_for_layer_load(next(iter(worker.context.layer_ids_by_name)))
        raise error
    assert raised.value is error
    assert connector._connector_metadata is None and worker._step is None
    assert not worker._executor.active
    with pytest.raises(ValueError) as closed:
        connector.shutdown()
    assert closed.value is error and resources.closed
    assert worker.take_released_store_job_ids() == {17}


def test_ready_store_on_no_forward_step_reports_source_release_without_wait_for_save(monkeypatch):
    worker, resources, backend = make_worker()
    connector = connect(worker, monkeypatch)
    command = RangeStoreCommand("ready", 0, 4, ((1,),), (b"a",), 4, 17)
    step = KVTransferStep(store=StoreCommandBatch(source_ready_commands=(command,)))
    try:
        with runner.KVConnectorModelRunnerMixin._get_kv_connector_output(output(step), wait_for_save=False):
            pass
        worker.store._pending[0].completion.result(timeout=2)
        connector.handle_preemptions(KVTransferStep())
        with runner.KVConnectorModelRunnerMixin._get_kv_connector_output(output(), wait_for_save=False) as result:
            pass
        assert result.kv_connector_worker_meta.released_store_jobs == {17: 1}
        assert connector.build_connector_worker_meta() is None
        assert any(call[0] == "put" for call in backend.calls)
    finally:
        connector.shutdown()
    assert resources.closed


def test_source_event_failure_during_bind_seals_accepted_work_and_preserves_error(monkeypatch):
    error = RuntimeError("source event failed")
    worker, resources, backend = make_worker(source_ready_event_factory=Mock(side_effect=error))
    connector = connect(worker, monkeypatch)
    command = RangeStoreCommand("ready", 0, 4, ((1,),), (b"a",), 4, 17)
    with pytest.raises(RuntimeError) as raised:
        connector.bind_connector_metadata(KVTransferStep(store=StoreCommandBatch(source_ready_commands=(command,))))
    assert raised.value is error
    assert connector._connector_metadata is None and worker._step is None
    with pytest.raises(RuntimeError) as closed:
        connector.shutdown()
    assert closed.value is error and resources.closed
    assert worker.take_released_store_job_ids() == {17}
    assert not any(call[0] == "put" for call in backend.calls)


def test_sync_load_start_failure_aborts_even_before_model_runner_enters_finally(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True)
    connector = connect(worker, monkeypatch, layerwise=True)
    error = RuntimeError("load opening failed")
    monkeypatch.setattr(backend, "start_key_range_load", Mock(side_effect=error))
    step = KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),))
    with (
        pytest.raises(RuntimeError) as raised,
        runner.KVConnectorModelRunnerMixin._get_kv_connector_output(output(step, sync=True)),
    ):
        pytest.fail("Forward must not run after Load startup failed")
    assert raised.value is error
    assert connector._connector_metadata is None and worker._step is None
    assert not worker._executor.active
    with pytest.raises(RuntimeError) as closed:
        connector.shutdown()
    assert closed.value is error and resources.closed
