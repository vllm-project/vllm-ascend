# SPDX-License-Identifier: Apache-2.0
"""Host recorder tests: no torch, vLLM, NPU, or global UT mocks required."""

import ast
import dataclasses
import json
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vllm_ascend.dfx.config import DfxConfig
from vllm_ascend.dfx.detectors import OutputContext, OutputDetectors, check_block_plan, check_positions
from vllm_ascend.dfx.inspect_trace import read_trace
from vllm_ascend.dfx.recorder import FlightRecorder, V2FlightRecorder


@dataclasses.dataclass
class Schedule:
    num_scheduled_tokens: dict[str, int] = dataclasses.field(default_factory=lambda: {"b": 2, "a": 1})
    total_num_scheduled_tokens: int = 3
    scheduled_spec_decode_tokens: dict = dataclasses.field(default_factory=dict)
    finished_req_ids: set = dataclasses.field(default_factory=set)
    connector: object = None


class HostTensor:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        return HostTensor(self.array[key])

    def numpy(self):
        return self.array


def runner():
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a", "b"], num_reqs=2, num_computed_tokens_cpu=np.array([9, 12])),
        input_ids=SimpleNamespace(cpu=HostTensor(np.array([42, 80, 90], dtype=np.int32))),
        requests={"a": SimpleNamespace(block_ids=([1, 2],)), "b": SimpleNamespace(block_ids=([3],))},
        use_async_scheduling=True,
        speculative_config=None,
        use_dcp=False,
    )


@pytest.fixture
def recorder(tmp_path):
    value = FlightRecorder.create(DfxConfig(enabled=True, output_dir=str(tmp_path)), {"dp_rank": 0, "rank": 0})
    assert value is not None
    yield value
    value.close()


def exported(recorder, reason="test", source=None):
    result = recorder.trigger(reason, source)
    assert result["status"] == "queued"
    recorder.close(timeout=5)
    assert recorder.stats()["dump_errors"] == 0
    return read_trace(Path(result["path"]))


@pytest.mark.parametrize("device_interval", [0, 1, 100])
def test_disabled_creates_no_thread_or_directory(tmp_path, device_interval):
    with patch("vllm_ascend.dfx.recorder.threading.Thread") as thread:
        value = FlightRecorder.create(
            DfxConfig(
                enabled=False,
                output_dir=str(tmp_path / "unused"),
                device_capture_interval=device_interval,
                detect_host_kv=True,
                track_block_ownership=True,
                audit_transfer_registration=True,
            ),
            {},
        )
    assert value is None
    thread.assert_not_called()
    assert not (tmp_path / "unused").exists()


def test_dfx_default_switches_are_off():
    config = DfxConfig()
    assert config.enabled is False
    assert config.device_capture_interval == 0


def v2_schedule(*, new=(), cached=(), blocks=(), resumed=(), finished=()):
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        finished_req_ids=set(finished),
        preempted_req_ids=set(),
        num_scheduled_tokens={"r": 1},
        total_num_scheduled_tokens=1,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(cached),
            new_block_ids=list(blocks),
            resumed_req_ids=set(resumed),
            num_computed_tokens=[3] * len(cached),
        ),
    )


def test_v2_block_history_append_resume_finish_and_budget(tmp_path):
    recorder = V2FlightRecorder(DfxConfig(enabled=True, output_dir=str(tmp_path), max_tracked_blocks=3), {})
    try:
        req = SimpleNamespace(req_id="r", block_ids=([1],), num_computed_tokens=0)
        recorder.start(v2_schedule(new=[req]))
        req.block_ids[0][0] = 99
        assert recorder.requests["r"].block_ids == ((1,),)
        recorder.start(v2_schedule(cached=["r"], blocks=[([2],)]))
        assert recorder.requests["r"].block_ids == ((1, 2),)
        recorder.start(v2_schedule(cached=["r"], blocks=[([4],)], resumed=["r"]))
        assert recorder.requests["r"].block_ids == ((4,),)
        recorder.start(v2_schedule(cached=["r"], blocks=[([5, 6, 7],)]))
        assert not recorder.requests
        recorder.start(v2_schedule(new=[req]))
        recorder.start(v2_schedule(finished=["r"]))
        assert not recorder.requests
    finally:
        recorder.close()


def test_v2_batch_keeps_device_tokens_unknown_and_copies_row_mapping(tmp_path):
    recorder = V2FlightRecorder(DfxConfig(enabled=True, output_dir=str(tmp_path)), {"runner": "mrv2"})
    recorder.start(v2_schedule(new=[SimpleNamespace(req_id="r", block_ids=([1],), num_computed_tokens=2)]))
    batch = SimpleNamespace(
        req_ids=["r"],
        num_reqs=1,
        num_scheduled_tokens=np.array([1]),
        idx_mapping_np=np.array([7]),
        query_start_loc_np=np.array([0, 1]),
        num_computed_tokens_np=np.array([2]),
    )
    recorder.batch(SimpleNamespace(pcp_manager=None, use_dcp=False), batch, SimpleNamespace(cg_mode="NONE"))
    batch.idx_mapping_np[0] = 9
    trace = exported(recorder)
    payload = next(record["payload"] for record in trace["records"] if record["kind"] == "host_batch_prepared")
    assert payload["input_token_ids"] is None
    assert payload["idx_mapping"] == [7]
    assert payload["actual_device_inputs_verified"] is False


def test_v2_finish_seals_frame_and_clears_active_state(tmp_path):
    recorder = V2FlightRecorder(DfxConfig(enabled=True, output_dir=str(tmp_path)), {})
    calls = []
    recorder.device_probe = SimpleNamespace(end=lambda frame, logits: calls.append(frame), counters={})
    recorder.frame = "owned_frame"
    recorder.active_execution = True
    recorder.finish()
    assert calls == ["owned_frame"]
    assert recorder.frame is None and not recorder.active_execution
    recorder.device_probe = None
    recorder.close()


@pytest.mark.parametrize(
    "dummy,profile,sync,enabled,raises",
    [
        (False, False, False, True, False),
        (False, False, False, True, True),
        (True, True, False, True, False),
        (True, False, False, True, False),
        (True, False, True, True, False),
        (False, False, False, False, False),
    ],
)
def test_v2_execute_hook_calls_parent_once_and_excludes_warmup(tmp_path, dummy, profile, sync, enabled, raises):
    # Execute the real override body against a CPU parent; no mocked rewrite of
    # its control flow and no dependency on vLLM/torch_npu imports.
    path = Path(__file__).parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "execute_model")
    method.decorator_list = []
    cls.body = [method]
    calls = []

    class Parent:
        def execute_model(self, *args, **kwargs):
            calls.append(kwargs)
            if raises:
                raise RuntimeError("original forward failed")
            return "original_output"

    namespace = dict(
        GPUModelRunner=Parent,
        torch=SimpleNamespace(),
        SchedulerOutput=object,
        IntermediateTensors=object,
        pcp_dispatch_context=nullcontext,
        _start_profiling_chunk_timing=lambda *args: None,
        _finish_profiling_chunk_timing=lambda *args: None,
        has_kv_transfer_group=lambda: False,
        vllm_version_is=lambda version: False,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    model = namespace["NPUModelRunner"]()
    model.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))
    model.model_state = SimpleNamespace()
    model.kvpp = SimpleNamespace(complete_forward=lambda: None)
    model._dfx_sync_only = sync
    recorder = V2FlightRecorder(DfxConfig(enabled=True, output_dir=str(tmp_path)), {}) if enabled else None
    model.dfx_recorder = recorder
    try:
        if raises:
            with pytest.raises(RuntimeError, match="original forward failed"):
                model.execute_model(v2_schedule(), dummy_run=dummy, is_profile=profile)
        else:
            assert model.execute_model(v2_schedule(), dummy_run=dummy, is_profile=profile) == "original_output"
        assert len(calls) == 1
        if recorder is not None:
            assert recorder.execution_id == int(not profile and (not dummy or sync))
            assert recorder.active_execution is False
    finally:
        if recorder is not None:
            recorder.close()


def test_v2_async_observer_does_not_resolve_early_or_record_twice():
    path = Path(__file__).parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    node = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "_DfxV2AsyncOutput"
    )
    namespace = {"AsyncModelRunnerOutput": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    result = object()
    inner = SimpleNamespace(model_runner_output=result, get_output=lambda: calls.append("resolve") or result)
    observer = namespace["_DfxV2AsyncOutput"](
        inner, SimpleNamespace(record_output=lambda output, context: calls.append(context)), "source_step"
    )
    assert not calls
    assert observer.get_output() is result
    assert observer.get_output() is result
    assert calls == ["resolve", "source_step", "resolve"]


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": True},
        {"enabled": True, "output_dir": " "},
        {"max_records": 0},
        {"max_record_bytes": 4095},
        {"max_buffer_bytes": 4096, "max_record_bytes": 8192},
        {"max_dumps": 0},
        {"run_id": "x" * 129},
        {"unknown_option": True},
    ],
)
def test_config_rejects_invalid_limits(values):
    with pytest.raises(ValueError):
        DfxConfig(**values)


def test_schedule_snapshot_owns_nested_mutable_values(recorder):
    schedule = Schedule(scheduled_spec_decode_tokens={"a": [7, 8]})
    recorder.record_schedule(schedule)
    schedule.scheduled_spec_decode_tokens["a"][0] = 999
    schedule.num_scheduled_tokens.clear()
    trace = exported(recorder)
    payload = trace["records"][0]["payload"]
    assert payload["scheduled_spec_decode_tokens"] == {"a": [7, 8]}
    assert payload["num_scheduled_tokens"] == {"b": 2, "a": 1}


def test_batch_records_execution_order_not_scheduler_dict_order(recorder):
    schedule = Schedule()
    model = runner()
    positions = np.array([9, 12, 13])
    recorder.record_schedule(schedule)
    recorder.record_batch(model, schedule, positions, np.array([1, 2]))
    positions[:] = -1
    model.input_ids.cpu.array[:] = -1
    model.requests["a"].block_ids[0].clear()
    model.input_batch.req_ids.reverse()
    trace = exported(recorder)
    record = trace["records"][1]
    assert record["violations"] == []
    assert record["payload"]["req_ids"] == ["a", "b"]
    assert record["payload"]["input_token_ids"] == [42, 80, 90]
    assert record["payload"]["logical_positions"] == [9, 12, 13]
    assert record["payload"]["block_ids"]["a"] == [[1, 2]]
    assert record["payload"]["actual_device_inputs_verified"] is False
    assert trace["coverage"]["exact_replay"] is False


def test_device_like_object_never_read_or_represented(recorder):
    class DeviceLike:
        @property
        def cpu(self):
            raise AssertionError("must not transfer")

        def __repr__(self):
            raise AssertionError("must not repr")

    recorder.record_schedule(Schedule(connector=DeviceLike()))
    record = exported(recorder)["records"][0]
    assert record["complete"] is False
    assert record["omissions"] == ["payload.connector"]


def test_oversized_record_leaves_explicit_gap(tmp_path):
    recorder = FlightRecorder(DfxConfig(output_dir=str(tmp_path), max_record_bytes=4096), {})
    recorder.record_schedule(Schedule(connector=list(range(10000))))
    trace = exported(recorder)
    assert trace["records"][0]["payload"] is None
    assert trace["records"][0]["omissions"] == ["record_budget_exceeded"]
    assert trace["stats"]["oversized_records"] == 1


def test_ring_eviction_exposes_retained_sequence_and_gap_count(tmp_path):
    recorder = FlightRecorder(DfxConfig(output_dir=str(tmp_path), max_records=2), {})
    for _ in range(5):
        recorder.record_schedule(Schedule())
    trace = exported(recorder, source=2)
    assert [record["execution_id"] for record in trace["records"]] == [4, 5]
    assert trace["stats"]["evicted_records"] == 3
    assert trace["trigger"]["source_execution_id"] == 2
    assert trace["trigger"]["capture_execution_id"] == 5


def test_byte_budget_evicts_before_record_count_limit(tmp_path):
    config = DfxConfig(output_dir=str(tmp_path), max_buffer_bytes=8192, max_record_bytes=8192)
    recorder = FlightRecorder(config, {})
    for _ in range(10):
        recorder.record_schedule(Schedule())
    assert recorder.stats()["accounted_buffer_bytes"] <= 8192
    assert recorder.stats()["evicted_records"] > 0
    recorder.close()


def test_schedule_violation_automatically_exports(recorder):
    recorder.record_schedule(Schedule(total_num_scheduled_tokens=4))
    recorder.close(timeout=5)
    paths = list(recorder.directory.glob("incident-*.json"))
    assert len(paths) == 1
    trace = read_trace(paths[0])
    assert trace["records"][0]["violations"] == ["schedule_total_mismatch"]


def test_batch_mapping_violation_automatically_exports(recorder):
    schedule = Schedule()
    recorder.record_schedule(schedule)
    recorder.record_batch(runner(), schedule, np.array([9, 12, 13]), np.array([2, 1]))
    recorder.close(timeout=5)
    trace = read_trace(next(recorder.directory.glob("incident-*.json")))
    assert trace["records"][-1]["violations"] == ["batch_count_mapping_mismatch"]


def test_capture_failure_does_not_escape(recorder):
    recorder.record_schedule(SimpleNamespace())
    recorder.record_schedule(Schedule())
    trace = exported(recorder)
    assert trace["stats"]["capture_errors"] == 1
    assert trace["records"][0]["execution_id"] == 2


def test_dump_failure_does_not_escape_or_report_completion(recorder):
    recorder.record_schedule(Schedule())
    with patch("vllm_ascend.dfx.recorder.os.open", side_effect=OSError("disk full")):
        assert recorder.trigger("disk full")["status"] == "queued"
        recorder.close(timeout=5)
    assert recorder.stats()["dump_errors"] == 1
    assert recorder.stats()["dumps_completed"] == 0
    assert not list(recorder.directory.glob("*.json"))


def test_initialization_failure_is_fail_open(tmp_path):
    with patch.object(Path, "mkdir", side_effect=OSError("read only")):
        assert FlightRecorder.create(DfxConfig(enabled=True, output_dir=str(tmp_path)), {}) is None


def test_slow_writer_queue_never_blocks_capture(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    original = json.dump

    def slow_dump(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    recorder = FlightRecorder(DfxConfig(output_dir=str(tmp_path)), {})
    try:
        with patch("vllm_ascend.dfx.recorder.json.dump", side_effect=slow_dump):
            recorder.record_schedule(Schedule())
            assert recorder.trigger("first")["status"] == "queued"
            assert entered.wait(5)
            assert recorder.trigger("second")["status"] == "queued"
            assert recorder.trigger("third")["status"] == "busy"
            recorder.record_schedule(Schedule())
            assert recorder.execution_id == 2
            release.set()
            recorder.close(timeout=5)
    finally:
        release.set()
        recorder.close()


def test_dump_quota_is_a_per_worker_lifetime_limit(tmp_path):
    recorder = FlightRecorder(DfxConfig(output_dir=str(tmp_path), max_dumps=1), {})
    assert recorder.trigger("first")["status"] == "queued"
    assert recorder.trigger("second")["status"] == "quota_exceeded"
    recorder.close()


def test_rank_and_restart_isolation(tmp_path):
    config = DfxConfig(output_dir=str(tmp_path), run_id="shared-run")
    first = FlightRecorder(config, {"dp_rank": 0})
    second = FlightRecorder(config, {"dp_rank": 1})
    try:
        assert first.directory != second.directory
        assert first.identity["worker_epoch"] != second.identity["worker_epoch"]
        assert first.identity["run_id"] == second.identity["run_id"]
        assert first.identity["dp_rank"] != second.identity["dp_rank"]
    finally:
        first.close()
        second.close()


def test_request_filter_preserves_cobatch_and_schedule(recorder):
    schedule = Schedule()
    recorder.record_schedule(schedule)
    recorder.record_batch(runner(), schedule, np.array([9, 12, 13]), np.array([1, 2]))
    result = recorder.trigger("filter")
    recorder.close(timeout=5)
    trace = read_trace(Path(result["path"]), request_id="a")
    assert len(trace["records"]) == 2
    assert trace["records"][1]["payload"]["req_ids"] == ["a", "b"]


def test_reader_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema":"unknown"}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        read_trace(path)


def test_close_stops_accepting_capture_and_export(recorder):
    recorder.close()
    recorder.record_schedule(Schedule())
    assert recorder.stats()["records"] == 0
    assert recorder.trigger("after shutdown")["status"] == "closed"


def test_large_integer_token_and_block_lists_round_trip_compactly(recorder):
    tokens = list(range(4096))
    schedule = Schedule(scheduled_spec_decode_tokens={"a": tokens})
    recorder.record_schedule(schedule)
    tokens[0] = 12345
    trace = exported(recorder)
    assert trace["stats"]["oversized_records"] == 0
    assert trace["records"][0]["payload"]["scheduled_spec_decode_tokens"]["a"] == list(range(4096))


def test_partial_connector_does_not_discard_supported_schedule_fields(recorder):
    recorder.record_schedule(Schedule(connector=object()))
    record = exported(recorder)["records"][0]
    assert record["complete"] is False
    assert record["payload"]["total_num_scheduled_tokens"] == 3


def test_cyclic_metadata_is_explicitly_incomplete(recorder):
    cyclic = []
    cyclic.append(cyclic)
    recorder.record_schedule(Schedule(connector=cyclic))
    record = exported(recorder)["records"][0]
    assert record["complete"] is False
    assert record["omissions"]


def test_dump_on_violation_can_be_disabled(tmp_path):
    recorder = FlightRecorder(DfxConfig(output_dir=str(tmp_path), dump_on_violation=False), {})
    recorder.record_schedule(Schedule(total_num_scheduled_tokens=4))
    assert recorder.stats()["dumps_submitted"] == 0
    trace = exported(recorder)
    assert trace["records"][0]["violations"] == ["schedule_total_mismatch"]


def test_negative_and_duplicate_counts_are_detected(recorder):
    recorder.config.dump_on_violation = False
    recorder.record_schedule(Schedule(num_scheduled_tokens={"a": -1}, total_num_scheduled_tokens=-1))
    model = runner()
    model.input_batch.req_ids = ["a", "a"]
    recorder.record_batch(model, Schedule(), np.array([9, 12, 13]), np.array([1, 2]))
    trace = exported(recorder)
    assert trace["records"][0]["violations"] == ["negative_scheduled_count"]
    assert "duplicate_batch_request" in trace["records"][1]["violations"]


def output(tokens, req_ids=None, **kwargs):
    req_ids = req_ids or ["a"]
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req: i for i, req in enumerate(req_ids)},
        sampled_token_ids=tokens,
        num_nans_in_logits=kwargs.get("nans"),
        logprobs=kwargs.get("logprobs"),
    )


def codes(findings):
    return {item["code"] for item in findings}


def test_repeat_cross_step_and_request_isolation():
    detector = OutputDetectors(DfxConfig(repeat_min_count=2, token_patterns=[[7, 8]]))
    assert not detector.check(output([[7], [8]], ["a", "b"]), OutputContext(1, (("a", 1), ("b", 2)), 100))
    findings = detector.check(output([[8], [8]], ["a", "b"]), OutputContext(2, (("a", 1), ("b", 2)), 100))
    assert {(f["code"], f["req_id"]) for f in findings} == {("output_token_pattern", "a"), ("token_repeat", "b")}
    assert all(f["severity"] == "symptom" for f in findings)


def test_request_reuse_does_not_join_histories():
    detector = OutputDetectors(DfxConfig(repeat_min_count=2))
    assert not detector.check(output([[7]]), OutputContext(1, (("a", 1),), 100))
    assert not detector.check(output([[7]]), OutputContext(2, (("a", 2),), 100))


def test_reordered_rows_detected_before_attribution():
    detector = OutputDetectors(DfxConfig())
    result = detector.check(output([[7], [8]], ["b", "a"]), OutputContext(1, (("a", 1), ("b", 2)), 100))
    assert codes(result) == {"output_request_mapping"}
    assert not detector.histories


@pytest.mark.parametrize("tokens", [[-1], [100], [1.5]])
def test_invalid_token(tokens):
    detector = OutputDetectors(DfxConfig())
    assert codes(detector.check(output([tokens]), OutputContext(1, (("a", 1),), 100))) == {"sampled_token_out_of_range"}


def test_existing_nan_counts_trigger():
    detector = OutputDetectors(DfxConfig())
    assert codes(detector.check(output([[1]], nans={"a": 3}), OutputContext(1, (("a", 1),), 100))) == {"logits_nan"}


@pytest.mark.parametrize(
    "value,expected", [(-np.inf, set()), (np.inf, {"logprob_nonfinite"}), (np.nan, {"logprob_nonfinite"})]
)
def test_masked_negative_infinity_is_legal(value, expected):
    detector = OutputDetectors(DfxConfig())
    values = SimpleNamespace(logprobs=np.array([[-0.1, value]]), logprob_token_ids=np.array([[1, 2]]))
    logprobs = SimpleNamespace(slice_request=lambda row, count: values)
    result = detector.check(output([[1]], logprobs=logprobs), OutputContext(1, (("a", 1),), 100))
    assert codes(result) == expected


def test_history_bounded_and_stale_output_ignored():
    detector = OutputDetectors(DfxConfig(max_tracked_requests=1, repeat_min_count=2))
    detector.check(output([[7]]), OutputContext(3, (("a", 1),), 100))
    assert not detector.check(output([[7]]), OutputContext(2, (("a", 1),), 100))
    assert detector.out_of_order == 1
    detector.check(output([[7]], ["b"]), OutputContext(4, (("b", 2),), 100))
    assert len(detector.histories) == 1 and detector.evictions == 1


def test_delayed_result_exports_original_execution(tmp_path):
    recorder = FlightRecorder(DfxConfig(output_dir=str(tmp_path)), {"dp_rank": 3})
    schedule = SimpleNamespace(num_scheduled_tokens={"a": 1}, total_num_scheduled_tokens=1)
    try:
        recorder.record_schedule(schedule)
        context = recorder.output_context(["a"], 100)
        recorder.record_schedule(schedule)
        recorder.record_output(output([[1]], nans={"a": 1}), context)
    finally:
        recorder.close(timeout=5)
    trace = read_trace(next(recorder.directory.glob("*.json")))
    assert trace["trigger"]["source_execution_id"] == 1
    assert trace["trigger"]["capture_execution_id"] == 2
    assert trace["records"][-1]["execution_id"] == 1
    assert trace["coverage"]["sampled_outputs"] is True
    assert trace["identity"]["dp_rank"] == 3


def test_contention_drops_without_waiting_and_resets_generation(tmp_path):
    recorder = FlightRecorder(DfxConfig(output_dir=str(tmp_path)), {})
    old = recorder.output_context(["a"], 100)
    held, release = threading.Event(), threading.Event()

    def hold():
        with recorder._capture_lock:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(2)
    try:
        assert recorder.trigger("test")["status"] == "contended"
        recorder.record_schedule(SimpleNamespace())
        assert recorder.stats()["contention_drops"] == 2
    finally:
        release.set()
        thread.join(2)
    new = recorder.output_context(["a"], 100)
    recorder.close()
    assert new.requests != old.requests


def test_async_bridge_resolves_once_and_preserves_result():
    # Execute the production adapter alone, not the NPU runner/import graph.
    # This is not a substitute for the real runner integration test.
    path = Path(__file__).parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "_DfxAsyncModelRunnerOutput"
    )
    namespace = {"AsyncModelRunnerOutput": type("AsyncModelRunnerOutput", (), {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    result = output([[1]])

    def resolve():
        calls.append("existing_resolution")
        return result

    observer = namespace["_DfxAsyncModelRunnerOutput"](
        SimpleNamespace(get_output=resolve, model_runner_output=result),
        SimpleNamespace(record_output=lambda value, context: calls.append((value, context))),
        "context",
    )
    assert observer.model_runner_output is result
    assert observer.get_output() is result
    assert calls == ["existing_resolution", (result, "context")]


def test_positions():
    assert not check_positions(np.array([5, 6, 8]), [2, 1], [5, 8])
    assert check_positions(np.array([5, 7, 8]), [2, 1], [5, 8]) == ["position_alignment"]


@pytest.mark.parametrize(
    "table,expected,positions,code",
    [
        ([[1], [1]], {"a": [1], "b": [1]}, [0, 0], "kv_planned_write_collision"),
        ([[1], [2]], {"a": [2], "b": [1]}, [0, 0], "block_table_request_mapping"),
        ([[1], [2]], {"a": [1], "b": [2]}, [8, 0], "kv_write_out_of_allocated_range"),
    ],
)
def test_kv_plan_fault_injection(table, expected, positions, code):
    assert code in check_block_plan(["a", "b"], [1, 1], np.array(positions), np.array(table), [1, 1], 8, expected)


def test_shared_prefix_reads_not_reported_as_writes():
    assert not check_block_plan(
        ["a", "b"],
        [1, 1],
        np.array([8, 8]),
        np.array([[1, 2], [1, 3]]),
        [2, 2],
        8,
        {"a": [1, 2], "b": [1, 3]},
    )


def test_host_plan_hook_auto_exports_collision(recorder):
    recorder.config.detect_host_kv = True
    model = runner()
    model.use_async_scheduling = False
    model.input_batch.num_computed_tokens_cpu = np.array([0, 0])
    table = SimpleNamespace(
        is_mamba_group=False,
        is_circular=False,
        use_hybrid_blocks=False,
        dcp_world_size=1,
        get_numpy_array=lambda: np.array([[1], [1]]),
        num_blocks_per_row=np.array([1, 1]),
        block_size=8,
    )
    model.input_batch.block_table = SimpleNamespace(block_tables=[table])
    model.requests = {req: SimpleNamespace(block_ids=([1],)) for req in ["a", "b"]}
    schedule = Schedule(num_scheduled_tokens={"a": 1, "b": 1}, total_num_scheduled_tokens=2)
    recorder.record_schedule(schedule)
    recorder.record_batch(model, schedule, np.array([0, 0]), np.array([1, 1]))
    recorder.close(timeout=5)
    trace = read_trace(next(recorder.directory.glob("*.json")))
    assert "kv_planned_write_collision" in trace["trigger"]["reason"]


def test_unsupported_async_host_plan_is_explicitly_skipped(recorder):
    recorder.config.detect_host_kv = True
    recorder.record_batch(runner(), Schedule(), np.array([9, 12, 13]), np.array([1, 2]))
    assert recorder.stats()["host_check_skips"] == 1
    assert recorder.stats()["capture_errors"] == 0


def test_detector_disabled_creates_no_output_context(recorder):
    recorder.config.detect_outputs = False
    assert recorder.output_context(["a"], 100) is None
    recorder.record_output(output([[1]]), None)
    assert recorder.stats()["output_records"] == 0


def test_new_admission_resets_generation(recorder):
    first = recorder.output_context(["a"], 100)
    schedule = Schedule()
    schedule.scheduled_new_reqs = [SimpleNamespace(req_id="a")]
    recorder.record_schedule(schedule)
    second = recorder.output_context(["a"], 100)
    assert first.requests != second.requests


def test_output_gap_does_not_join_token_history(recorder):
    recorder.config.repeat_min_count = 2
    context = recorder.output_context(["a"], 100)
    recorder.record_output(output([[7]]), context)
    recorder._history_gap = True
    recorder.record_schedule(Schedule())
    recorder.record_output(output([[7]]), recorder.output_context(["a"], 100))
    assert recorder.stats()["detector_findings"] == 0


def test_output_check_budget_is_not_claimed_clean():
    detector = OutputDetectors(DfxConfig(max_checked_tokens=1))
    findings = detector.check(output([[1, 2]]), OutputContext(1, (("a", 1),), 100))
    assert findings[0]["code"] == "output_check_budget_exceeded"
    assert findings[0]["severity"] == "coverage_gap"


def test_output_capture_failure_preserves_serving(recorder):
    context = recorder.output_context(["a"], 100)
    recorder.record_output(SimpleNamespace(), context)
    assert recorder.stats()["capture_errors"] == 1
    assert recorder._history_gap


def test_spec_acceptance_by_request_not_current_row():
    detector = OutputDetectors(DfxConfig(spec_min_proposals=2, spec_acceptance_floor=0.2))
    context = OutputContext(1, (("b", 2), ("a", 1)), 100, (("a", 2), ("b", 2)))
    findings = detector.check(output([[7, 8, 9], [1]], ["b", "a"]), context)
    assert {(f["code"], f["req_id"]) for f in findings} == {("spec_acceptance_low", "a")}
    assert findings[0]["severity"] == "symptom"


def test_spec_discarded_row_not_counted_as_rejection():
    detector = OutputDetectors(DfxConfig(spec_min_proposals=1))
    context = OutputContext(1, (("a", 1),), 100, (("a", 4),))
    assert not detector.check(output([[]]), context)
    assert not detector.acceptance


def test_device_bridge_reaches_probe(recorder):
    calls = []
    recorder.device_probe = SimpleNamespace(
        begin=lambda *args, **kwargs: calls.append((args, kwargs)) or "frame",
        end=lambda *args: calls.append(args),
        counters={},
        poll=lambda: None,
    )
    assert recorder.begin_device("runner", "metadata", "ids", "positions", kind="real") == "frame"
    recorder.end_device("frame", "logits")
    assert calls == [(("runner", "metadata", "ids", "positions"), {"kind": "real"}), ("frame", "logits")]
    recorder.device_probe = None


def test_device_enabled_without_completed_evidence_not_claimed_checked(recorder):
    recorder.device_probe = SimpleNamespace(counters={}, poll=lambda: None, has_pending=lambda: False)
    trace = exported(recorder)
    assert trace["coverage"]["device_capture_enabled"] is True
    assert trace["coverage"]["kv_contents"] is False
    assert trace["coverage"]["raw_logits_finite_checked"] is False
    recorder.device_probe = None
