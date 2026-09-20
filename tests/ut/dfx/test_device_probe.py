# SPDX-License-Identifier: Apache-2.0
"""Device probe algorithms on real CPU tensors with a controlled event API.

These tests verify ownership/check logic, NOT Ascend stream or graph correctness.
"""

import time
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.dfx.config import DfxConfig
from vllm_ascend.dfx.detectors import BlockOwnership, check_transfer_registration
from vllm_ascend.dfx.device_probe import DeviceProbe
from vllm_ascend.dfx.inspect_trace import compare_device_snapshots, compare_traces, decode_tensor, read_trace
from vllm_ascend.dfx.recorder import FlightRecorder


@dataclass
class AscendMetadata:
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 2
    seq_lens: torch.Tensor | None = None


class Event:
    def __init__(self):
        self.ready = False
        self.recorded = False

    def record(self, stream):
        self.recorded = True

    def query(self):
        return self.ready

    def synchronize(self):
        raise AssertionError("DFX must not synchronize")


class API:
    def __init__(self):
        self.events = []

    def Event(self):
        event = Event()
        self.events.append(event)
        return event

    def current_stream(self):
        return None


def setup_probe(**options):
    events = []
    config = DfxConfig(device_capture_interval=1, **options)
    recorder = SimpleNamespace(config=config, execution_id=1, record_event=lambda *args: events.append(args))
    table = SimpleNamespace(
        block_size=4,
        is_circular=False,
        use_hybrid_blocks=False,
        num_blocks_per_row=np.array([1]),
        get_numpy_array=lambda: np.array([[1]]),
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer"])]),
        input_batch=SimpleNamespace(req_ids=["req"], num_reqs=1, block_table=SimpleNamespace(block_tables=[table])),
        use_dcp=False,
        speculative_config=None,
        requests={"req": SimpleNamespace(prompt_token_ids=[10, 11], output_token_ids=[], lora_request=None)},
    )
    caches = {"layer": (torch.zeros(3, 1, 4, 2), torch.zeros(3, 1, 4, 2))}
    api = API()
    probe = DeviceProbe(recorder, runner, caches, api, cache_layout="HND")
    metadata = {"layer": AscendMetadata(torch.tensor([4, 5]), torch.tensor([[1]]), torch.tensor([0, 2]))}
    return probe, runner, caches, metadata, api, events


def begin(probe, runner, metadata, **kwargs):
    return probe.begin(runner, metadata, torch.tensor([10, 11]), torch.tensor([0, 1]), **kwargs)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_v2_device_inputs_slots_and_kv_are_captured(layout):
    probe, runner, caches, _, api, events = setup_probe()
    probe.cache_layout = layout
    if layout == "NHD":
        caches["layer"] = tuple(t.movedim(1, 2) for t in caches["layer"])
    runner.block_tables = SimpleNamespace(kernel_block_sizes=[4], block_sizes=[4])
    runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec = type("FullAttentionSpec", (), {})()
    runner.pcp_manager = None
    runner.kvpp = SimpleNamespace(scheduler=None)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    batch = SimpleNamespace(
        req_ids=["req"],
        num_reqs=1,
        input_ids=torch.tensor([10, 11]),
        positions=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
        num_tokens=2,
    )
    requests = {"req": SimpleNamespace(block_ids=([1],))}
    frame = probe.begin_v2(
        runner, batch, (torch.tensor([[1]]),), torch.tensor([[4, 4]]), requests, kind="real", graph_mode="NONE"
    )
    assert frame is not None and frame["blocks"] == (1,)
    assert not frame["request_context_complete"]
    batch.input_ids.fill_(99)
    finish(probe, frame, api)
    payload, findings = events[-1][1:3]
    assert decode_tensor(payload["tensors"]["input_ids"]).tolist() == [10, 11]
    assert "device_slot_write_collision" in findings
    assert "device_position_slot_mismatch" in findings
    assert "kv.after.0.1" in payload["tensors"]


def test_v2_probe_ignores_unjoined_cp_streams():
    probe, runner, _, _, api, events = setup_probe()
    runner.block_tables = SimpleNamespace(kernel_block_sizes=[4], block_sizes=[4])
    runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec = object()
    runner.pcp_manager = object()
    runner.kvpp = SimpleNamespace(scheduler=None)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    batch = SimpleNamespace(
        req_ids=[], num_reqs=0, query_start_loc=torch.tensor([0]), seq_lens=torch.tensor([]), num_tokens=0
    )
    assert (
        probe.begin_v2(runner, batch, (torch.tensor([[]]),), torch.tensor([[]]), {}, kind="real", graph_mode="NONE")
        is None
    )
    assert not api.events
    assert events[-1][1]["reason"] == "v2_cp_or_ubatch_stream_ordering_unverified"


def finish(probe, frame, api, logits=None):
    probe.end(frame, logits)
    for event in api.events:
        event.ready = True
    probe.poll()


def test_snapshot_is_owned_and_waits_for_existing_event_query():
    probe, runner, caches, metadata, api, events = setup_probe()
    frame = begin(probe, runner, metadata)
    caches["layer"][0][1, 0, 0] = 12
    probe.end(frame)
    caches["layer"][0].fill_(99)
    metadata["layer"].slot_mapping.fill_(100)
    probe.poll()
    assert len(events) == 1  # Layout only; pending buffers have not been inspected.
    api.events[0].ready = True
    probe.poll()
    assert events[-1][0] == "device_snapshot"
    tensor = events[-1][1]["tensors"]["kv.after.0.1"]
    values = np.frombuffer(tensor["data"].tobytes(), dtype=np.float32).reshape(1, 4, 2)
    assert values[0, 0, 0] == 12 and values[0, 2, 0] == 0
    assert events[-1][2] == ()


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("collision", "device_slot_write_collision"),
        ("range", "device_slot_out_of_range"),
        ("mapping", "device_position_slot_mismatch"),
        ("offsets", "device_query_offsets_invalid"),
        ("padding", "device_padding_writable_slots"),
    ],
)
def test_metadata_fault_injection(fault, expected):
    probe, runner, _, metadata, api, events = setup_probe()
    value = metadata["layer"]
    if fault == "collision":
        value.slot_mapping = torch.tensor([4, 4])
    elif fault == "range":
        value.slot_mapping = torch.tensor([4, 200])
    elif fault == "mapping":
        value.slot_mapping = torch.tensor([5, 4])
    elif fault == "offsets":
        value.query_start_loc = torch.tensor([1, 0])
    else:
        value.slot_mapping = torch.tensor([4, 5, 0])
    finish(probe, begin(probe, runner, metadata), api)
    assert expected in events[-1][2]


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("write", "kv_changed_outside_planned_slots"),
        ("nan", "written_kv_nonfinite"),
        ("logits", "raw_logits_nonfinite"),
    ],
)
def test_data_fault_injection(fault, expected):
    probe, runner, caches, metadata, api, events = setup_probe()
    frame = begin(probe, runner, metadata)
    if fault == "write":
        caches["layer"][0][1, 0, 3] = 42
    elif fault == "nan":
        caches["layer"][0][1, 0, 0] = float("nan")
    finish(probe, frame, api, torch.tensor([[float("nan")]]) if fault == "logits" else None)
    assert expected in events[-1][2]


def test_invalid_tail_nan_not_treated_as_valid_kv_nan():
    probe, runner, caches, metadata, api, events = setup_probe()
    caches["layer"][0][1, 0, 3] = float("nan")
    finish(probe, begin(probe, runner, metadata), api)
    assert not events[-1][2]
    assert events[-1][1]["tensors"]["kv.after.0.1"]["nonfinite_elements_including_unused"] == 2


def test_existing_sync_forward_writes_detected_without_running_a_model():
    probe, runner, caches, metadata, api, events = setup_probe()
    frame = begin(probe, runner, metadata, kind="sync_only")
    caches["layer"][0][0].fill_(1)
    finish(probe, frame, api)
    assert {"sync_only_writable_slots", "sync_only_kv_changed"} <= set(events[-1][2])


def test_capture_and_profile_are_skipped():
    probe, runner, _, metadata, api, _ = setup_probe()
    assert begin(probe, runner, metadata, skip=True) is None
    assert not api.events


def test_graph_capture_does_not_claim_eager_metadata_validation():
    probe, runner, _, metadata, api, events = setup_probe()
    finish(probe, begin(probe, runner, metadata, graph_mode="FULL"), api)
    assert events[-1][1]["simple"] is False


def test_pending_queue_is_bounded_and_never_waits():
    probe, runner, _, metadata, api, _ = setup_probe(device_pending_limit=1)
    probe.end(begin(probe, runner, metadata))
    assert begin(probe, runner, metadata) is None
    assert probe.counters["busy"] == 1
    assert len(probe.pending) == 1 and not api.events[0].ready


def test_oversized_tensor_is_omitted_without_copying():
    probe, runner, _, metadata, api, events = setup_probe(device_capture_bytes=4096)
    metadata["layer"].block_tables = torch.zeros(10000, 2, dtype=torch.int64)
    finish(probe, begin(probe, runner, metadata), api)
    assert "metadata.block_tables" in events[-1][1]["omissions"]
    assert "metadata.block_tables" not in events[-1][1]["tensors"]


def test_strided_recurrent_components_remain_separate():
    probe, runner, caches, metadata, api, events = setup_probe()
    storage = torch.arange(48, dtype=torch.float32)
    caches["layer"] = (torch.as_strided(storage, (3, 2), (16, 1), 0), torch.as_strided(storage, (3, 2), (16, 1), 8))
    metadata = {"layer": SimpleNamespace()}
    finish(probe, begin(probe, runner, metadata), api)
    tensors = events[-1][1]["tensors"]
    assert np.frombuffer(tensors["kv.after.0.1"]["data"], dtype=np.float32).tolist() == [16, 17]
    assert np.frombuffer(tensors["kv.after.1.1"]["data"], dtype=np.float32).tolist() == [24, 25]


def test_bfloat16_preserved_as_raw_bytes():
    probe, runner, caches, metadata, api, events = setup_probe()
    caches["layer"] = (torch.ones(3, 1, 4, 2, dtype=torch.bfloat16),)
    finish(probe, begin(probe, runner, metadata), api)
    tensor = events[-1][1]["tensors"]["kv.after.0.1"]
    assert tensor["layout"]["dtype"] == "torch.bfloat16"
    assert len(tensor["data"]) == 16


def test_block_ownership_reuse_and_shared_prefix():
    tracker = BlockOwnership(DfxConfig())
    request = lambda blocks, count=0: SimpleNamespace(block_ids=(blocks,), num_computed_tokens=count)
    first, _ = tracker.observe({"a": request([1, 2]), "b": request([1, 3])})
    assert first["changes"][0]["owners"] == ["a", "b"]
    removed, _ = tracker.observe({})
    assert all(event["event"] == "detach" for event in removed["changes"])
    reused, _ = tracker.observe({"c": request([1])})
    assert reused["changes"][0]["association_epoch"] > first["changes"][0]["association_epoch"]
    assert reused["authority"] == "worker_observed_not_allocator"


def test_progress_rollback_and_resume():
    tracker = BlockOwnership(DfxConfig())
    request = SimpleNamespace(block_ids=([1],), num_computed_tokens=10)
    tracker.observe({"a": request}, check_progress=True)
    request.num_computed_tokens = 5
    _, findings = tracker.observe({"a": request}, check_progress=True)
    assert findings == ["computed_tokens_rollback_without_resume"]
    request.num_computed_tokens = 0
    assert not tracker.observe({"a": request}, check_progress=True, allow_rollback={"a"})[1]


def test_ownership_budget_clears_incomplete_baseline():
    tracker = BlockOwnership(DfxConfig(max_tracked_blocks=1))
    payload, _ = tracker.observe({"a": SimpleNamespace(block_ids=([1, 2],), num_computed_tokens=0)})
    assert payload["complete"] is False and not tracker.previous


def snapshot():
    probe, runner, _, metadata, api, events = setup_probe()
    finish(probe, begin(probe, runner, metadata), api)
    return events[-1][1]


def test_reference_comparison_detects_finite_wrong_kv():
    reference = snapshot()
    actual = deepcopy(reference)
    tensor = actual["tensors"]["kv.after.0.1"]
    tensor["data"][:4] = np.frombuffer(np.float32(1.0).tobytes(), dtype=np.uint8)
    result = compare_device_snapshots(actual, reference)
    assert result["status"] == "mismatch"
    assert result["comparisons"]["kv.after.0.1"]["max_abs_error"] == 1.0


def test_reference_comparison_excludes_invalid_tail():
    reference = snapshot()
    actual = deepcopy(reference)
    tensor = actual["tensors"]["kv.after.0.1"]
    tensor["data"][-4:] = np.frombuffer(np.float32(float("nan")).tobytes(), dtype=np.uint8)
    assert compare_device_snapshots(actual, reference)["status"] == "observed_match"


@pytest.mark.parametrize("field,value", [("req_ids", ["different"]), ("blocks", [2]), ("input_digest", "different")])
def test_reference_comparison_rejects_unaligned_evidence(field, value):
    reference = snapshot()
    actual = deepcopy(reference)
    actual[field] = value
    assert compare_device_snapshots(actual, reference)["status"] == "inconclusive"


def test_missing_reference_tensor_is_not_a_pass():
    reference = snapshot()
    actual = deepcopy(reference)
    del actual["tensors"]["kv.after.0.1"]
    assert compare_device_snapshots(actual, reference)["status"] == "inconclusive"


def test_bfloat16_comparison_decode():
    values = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    payload = {"layout": {"dtype": "torch.bfloat16", "shape": [2]}, "data": values.view(torch.uint8).numpy()}
    assert decode_tensor(payload).tolist() == [1.0, -2.0]


def test_online_reference_mismatch_produces_finding():
    probe, runner, caches, metadata, api, events = setup_probe()
    manifest = {"model": "test", "revision": "test", "parallel": {"tensor_parallel_size": 1}}
    identity = {"rank": 0, "dp_rank": 0}
    probe.recorder._manifest = manifest
    probe.recorder.identity = identity
    probe.reference = {
        "manifest": manifest,
        "identity": identity,
        "records": [
            {"kind": "device_snapshot", "execution_id": 1, "payload": snapshot()},
        ],
    }
    frame = begin(probe, runner, metadata)
    caches["layer"][0][1, 0, 0] = 1
    finish(probe, frame, api)
    assert "reference_numeric_mismatch" in events[-1][2]


def test_reference_requires_model_and_rank_alignment():
    record = {"kind": "device_snapshot", "execution_id": 1, "payload": snapshot()}
    trace = {
        "manifest": {"model": "test", "revision": "rev", "parallel": {}},
        "identity": {"rank": 0},
        "records": [record],
    }
    other = deepcopy(trace)
    other["identity"]["rank"] = 1
    assert compare_traces(trace, other)["reason"] == "rank_mismatch"


def test_nhd_token_axis_uses_explicit_layout():
    probe, runner, caches, metadata, api, events = setup_probe()
    probe.cache_layout = "NHD"
    caches["layer"] = (torch.zeros(3, 4, 1, 2),)
    frame = begin(probe, runner, metadata)
    caches["layer"][0][1, 3] = 2
    finish(probe, frame, api)
    assert "kv_changed_outside_planned_slots" in events[-1][2]


def test_unordered_external_writers_disable_kv_snapshot():
    probe, runner, _, metadata, api, events = setup_probe()
    runner.vllm_config = SimpleNamespace(kv_transfer_config=object())
    finish(probe, begin(probe, runner, metadata), api)
    payload = events[-1][1]
    assert payload["cache_safe"] is False
    assert not any(name.startswith("kv.") for name in payload["tensors"])
    assert "kv.before:external_writer_ordering_unverified" in payload["omissions"]


def test_event_record_failure_retains_buffers_and_stops_allocation():
    probe, runner, _, metadata, _, _ = setup_probe()
    frame = begin(probe, runner, metadata)
    probe.api.Event = lambda: (_ for _ in ()).throw(RuntimeError("injected event failure"))
    probe.end(frame)
    assert probe.disabled and probe.quarantine == [frame]
    assert begin(probe, runner, metadata) is None


@pytest.mark.parametrize(
    "length,stride,groups,expected",
    [
        (32, 64, [0, 1], []),
        (80, 64, [0, 1], ["transfer_block_length_exceeds_stride"]),
        (32, 32, [0, 1], ["transfer_stride_mismatch"]),
        (32, 64, [0], ["transfer_missing_alias_group"]),
    ],
)
def test_transfer_registration_contract(length, stride, groups, expected):
    views = {1000: dict(blocks=4, stride_bytes=64, storage_begin=1000, storage_end=1512, groups=[0, 1])}
    entries = [dict(base=1000, length=length, stride=stride, groups=groups)]
    assert check_transfer_registration(entries, views, 4) == expected


def test_transfer_logical_to_physical_scaling():
    views = {1000: dict(blocks=8, stride_bytes=64, storage_begin=1000, storage_end=1512, groups=[0])}
    assert not check_transfer_registration([dict(base=1000, length=128, stride=128)], views, 4)
    findings = check_transfer_registration([dict(base=1000, length=256, stride=128)], views, 4)
    assert "transfer_block_length_exceeds_stride" in findings
    assert "transfer_registration_out_of_allocation" in findings


def test_valid_prefix_nan_detected_even_when_not_written():
    probe, runner, caches, metadata, api, events = setup_probe()
    metadata["layer"].seq_lens = torch.tensor([4])
    caches["layer"][0][1, 0, 3] = float("nan")
    finish(probe, begin(probe, runner, metadata), api)
    assert "valid_kv_nonfinite" in events[-1][2]
    assert "written_kv_nonfinite" not in events[-1][2]


@dataclass
class AscendMLAMetadata(AscendMetadata):
    pass


def test_mla_uses_token_major_layout_even_with_hnd_setting():
    probe, runner, caches, _, api, events = setup_probe()
    caches["layer"] = (torch.zeros(3, 4, 1, 2), torch.zeros(3, 4, 1, 2))
    metadata = {"layer": AscendMLAMetadata(torch.tensor([4, 5]), torch.tensor([[1]]), torch.tensor([0, 2]))}
    frame = begin(probe, runner, metadata)
    caches["layer"][0][1, 3] = 2
    finish(probe, frame, api)
    assert "kv_changed_outside_planned_slots" in events[-1][2]


@dataclass
class GDNAttentionMetadata:
    non_spec_state_indices_tensor: torch.Tensor
    non_spec_query_start_loc: torch.Tensor


@dataclass
class ConvMetadata:
    cache_indices: torch.Tensor
    query_start_loc: torch.Tensor


@dataclass
class GDNDecodeMetadata:
    causal_conv1d: ConvMetadata


@pytest.mark.parametrize("component", [0, 1])
def test_gdn_conv_and_ssm_finite_check(component):
    probe, runner, caches, _, api, events = setup_probe()
    caches["layer"] = (torch.zeros(3, 2), torch.zeros(3, 2))
    value = GDNAttentionMetadata(torch.tensor([1]), torch.tensor([0, 1]))
    value.non_spec_decode_metadata = GDNDecodeMetadata(ConvMetadata(torch.tensor([1]), torch.tensor([0, 1])))
    frame = begin(probe, runner, {"layer": value})
    caches["layer"][component][1] = float("nan")
    finish(probe, frame, api)
    assert "recurrent_state_nonfinite" in events[-1][2]
    assert events[-1][1]["tensors"][f"kv.after.{component}.1"]["whole_state_valid"]


def test_gdn_zero_length_query_must_not_change_state():
    probe, runner, caches, _, api, events = setup_probe()
    caches["layer"] = (torch.zeros(3, 2), torch.zeros(3, 2))
    value = GDNAttentionMetadata(torch.tensor([1]), torch.tensor([0, 0]))
    frame = begin(probe, runner, {"layer": value})
    caches["layer"][1][1] = 12
    finish(probe, frame, api)
    assert "recurrent_state_changed_without_active_query" in events[-1][2]


def test_actual_graph_capture_state_prevents_probe():
    probe, runner, _, metadata, api, _ = setup_probe()
    probe.device = SimpleNamespace(type="npu")
    api.is_current_stream_capturing = lambda: True
    assert begin(probe, runner, metadata) is None
    assert not probe.pending and probe.counters["skipped"] == 1


def test_unjoined_metadata_producer_is_not_read():
    probe, runner, _, metadata, api, events = setup_probe()
    runner.device_metadata_executor = SimpleNamespace(submission_in_flight=True)
    assert begin(probe, runner, metadata) is None
    assert not api.events
    assert events[-1][0] == "device_capture_gap"


def test_background_detects_final_idle_step_and_exports_real_bytes(tmp_path):
    probe, runner, caches, metadata, api, _ = setup_probe()
    probe.config.output_dir = str(tmp_path)
    recorder = FlightRecorder(probe.config, {"rank": 0, "dp_rank": 0})
    probe.recorder = recorder
    recorder.device_probe = probe
    try:
        frame = begin(probe, runner, metadata)
        caches["layer"][0][1, 0, 0] = float("nan")
        probe.end(frame)
        api.events[0].ready = True
        deadline = time.monotonic() + 3
        while recorder.stats()["dumps_completed"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert recorder.stats()["dumps_completed"] == 1
        assert recorder.stats()["device_errors"] == 0
    finally:
        recorder.close(timeout=3)
    trace = read_trace(next(recorder.directory.glob("*.json")))
    record = next(item for item in trace["records"] if item["kind"] == "device_snapshot")
    assert "written_kv_nonfinite" in record["violations"]
    assert np.isnan(decode_tensor(record["payload"]["tensors"]["kv.after.0.1"])[0, 0, 0])
    assert trace["coverage"]["kv_contents"] == "sampled_pages"
