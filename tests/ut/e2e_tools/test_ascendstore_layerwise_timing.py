"""Verify that the timing adapter preserves actual v1 Worker calls and execution results."""

from types import SimpleNamespace

from tests.e2e.pull_request.one_card.ascendstore_v1_layerwise_timing import LayerwiseLoadTiming
from tests.ut.distributed.ascend_store.v1.helpers import FakeLoadStartGate, make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    StoreCommandBatch,
)


def test_host_timing_preserves_real_layerwise_load_calls_and_captures_background_work():
    worker, resources, backend = make_worker(layerwise=True, start_gate_factory=lambda: FakeLoadStartGate(opened=True))
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))
    metadata = KVTransferStep((command,), StoreCommandBatch(()))
    schedule = SimpleNamespace(
        kv_connector_metadata=metadata, num_scheduled_tokens={"request": 1}, total_num_scheduled_tokens=1
    )
    lookup = SimpleNamespace(_lookup=worker.lookup)
    lookup_request = LookupRequest(0, 4, (0,), (b"a",))

    def execute_model(scheduler_output):
        worker.begin_step(scheduler_output.kv_connector_metadata)
        try:
            worker.start_load()
            worker.wait_for_layer_load("layers.0.group.0")
            worker.wait_for_layer_load("layers.1.group.0")
            result = worker.collect_load_result()
            assert not result.failed_locations
            return result
        finally:
            worker.end_step()

    runner = SimpleNamespace(execute_model=execute_model, sample_tokens=lambda grammar_output: "token")
    timing = LayerwiseLoadTiming(worker, backend, lookup, runner)
    original_copy = worker.transfer._load_ranges
    try:
        baseline_start = len(backend.calls)
        baseline_lookup = lookup._lookup(lookup_request)
        baseline_result = runner.execute_model(schedule)
        baseline_calls = backend.calls[baseline_start:]
        timing.install()
        try:
            observed_start = len(backend.calls)
            assert lookup._lookup(lookup_request) == baseline_lookup
            assert runner.execute_model(schedule) == baseline_result
            assert runner.sample_tokens(None) == "token"
            assert backend.calls[observed_start:] == baseline_calls
        finally:
            timing.restore()
        report = timing.report()
        assert report["failed_calls"] == report["dropped_records"] == 0
        assert report["stages"]["lookup_service"]["calls"] == 1
        assert report["stages"]["lookup_backend_exists"]["calls"] == 1
        step = report["steps"][0]
        assert step["stages"]["session_start"]["calls"] == step["stages"]["session_end"]["calls"] == 1
        assert step["stages"]["layer_plan_prepare"]["calls"] == 1
        assert step["stages"]["range_copy"]["calls"] == 2
        assert [layer["layer"] for layer in report["layers"]] == [0, 1]
        for layer in report["layers"]:
            for stage in ("layer_task_host", "layer_io_host", "range_copy", "layer_wait_host"):
                assert layer["stages"][stage]["calls"] == 1
        assert "start_load" not in vars(worker)
        assert worker.transfer._load_ranges is original_copy
        assert lookup._lookup == worker.lookup
    finally:
        timing.restore()
        worker.close()
    assert resources.closed
