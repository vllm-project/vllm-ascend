# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure CPU diagnostics tests, runnable with --confcutdir=tests/ut/debug.

Load the dependency-free modules directly so the tests also run without a
complete vLLM/NPU installation (the project's parent conftest requires one).
"""

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[3] / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trace_module = load_module("kv_trace_test_subject", "vllm_ascend/debug/kv_trace.py")
manager_module = load_module("kv_manager_test_subject", "vllm_ascend/debug/kv_trace_manager.py")
cli = load_module("kv_trace_cli_test_subject", "tools/kv_block_trace.py")


def writer(tmp_path, **kwargs):
    return trace_module.KVTrace(
        trace_module.TraceConfig(str(tmp_path), **kwargs), "test", engine_id="engine", dp_rank=0
    )


def events(tmp_path):
    return cli.load_events([tmp_path])


@pytest.mark.parametrize(
    "update",
    [
        {"directory": ""},
        {"max_events": 0},
        {"snapshots": "yes"},
        {"layers": []},
        {"run_id": "../escape"},
        {"unknown": 1},
    ],
)
def test_invalid_configuration(update):
    with pytest.raises(ValueError):
        trace_module.TraceConfig.parse(json.dumps({"directory": "/tmp/trace", **update}))


def test_writer_thread_safety_and_explicit_truncation(tmp_path):
    trace = writer(tmp_path, max_events=80)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: trace.emit("event", index=i), range(100)))
    rows = events(tmp_path)
    assert len(rows) == 81
    assert [r["sequence"] for r in rows] == list(range(1, 82))
    assert rows[-1]["event"] == "trace.truncated"
    assert len({r["trace_id"] for r in rows}) == 1


def test_io_failure_disables_observation(tmp_path):
    (tmp_path / "kv-trace").write_text("not a directory")
    trace = writer(tmp_path)
    trace.emit("event")
    assert not trace.enabled


def make_runner(block_size=8):
    cache = torch.zeros(3, block_size, 1, 4)
    table = NS(block_size=block_size, slot_mapping=NS(gpu=torch.tensor([block_size + 3])))
    table.get_device_tensor = lambda: torch.tensor([[1]])
    spec = NS(head_size=4, block_size=block_size)
    runner = NS(
        input_batch=NS(req_ids=[], block_table=NS(block_tables=[table])),
        attn_state="DecodeOnly",
        kv_cache_config=NS(kv_cache_groups=[NS(layer_names=["layer0"], kv_cache_spec=spec)]),
        _kv_trace_caches={"layer0": (cache,)},
    )
    return runner, cache


def test_dummy_snapshot_reports_actual_changed_position_without_modifying_inputs(tmp_path):
    trace = writer(tmp_path, snapshots=True)
    runner, cache = make_runner()
    slots = runner.input_batch.block_table.block_tables[0].slot_mapping.gpu.clone()

    def forward():
        cache[1, 3] = 7
        return "model output"

    observed = trace.wrap_forward(runner, forward, 1, "dummy", NS(capturing=False, cudagraph_runtime_mode="FULL"))
    assert observed() == "model output"
    assert torch.equal(slots, runner.input_batch.block_table.block_tables[0].slot_mapping.gpu)
    rows = events(tmp_path)
    diff = next(r for r in rows if r["event"] == "cache.diff")
    assert diff["block_size"] == 8
    assert diff["changed_offsets"] == [3]
    assert diff["changed_elements"] == 4
    assert diff["before_sha256"] != diff["after_sha256"]
    assert rows[-1]["completion"] == "device_observed"


def test_unchanged_nans_are_not_reported_as_writes():
    before = torch.tensor([[float("nan"), 0.0], [1.0, -0.0]])
    assert trace_module.compare_snapshot(before, before.clone())["changed_elements"] == 0
    after = before.clone()
    after[1, 1] = 0.0
    assert trace_module.compare_snapshot(before, after)["changed_offsets"] == [1]


@pytest.mark.parametrize("phase,capturing", [("warmup", False), ("forward", True)])
def test_profile_and_capture_do_not_observe_device_tensors(tmp_path, phase, capturing):
    trace = writer(tmp_path, snapshots=True)
    forward = lambda: 42
    assert trace.wrap_forward(None, forward, 1, phase, NS(capturing=capturing)) is forward
    assert not list(tmp_path.rglob("*.jsonl"))


def test_snapshot_budget_and_model_exception(tmp_path):
    trace = writer(tmp_path, snapshots=True, max_snapshot_bytes=1)
    runner, _ = make_runner()

    def forward():
        raise RuntimeError("real model failure")

    with pytest.raises(RuntimeError, match="real model failure"):
        trace.wrap_forward(runner, forward, 1, "dummy", NS(capturing=False, cudagraph_runtime_mode="FULL"))()
    rows = events(tmp_path)
    assert any(r["event"] == "snapshot.truncated" for r in rows)
    assert rows[-1]["event"] == "forward.error"


def test_metadata_only_does_not_snapshot(tmp_path):
    trace = writer(tmp_path)
    runner, _ = make_runner()
    del runner._kv_trace_caches
    assert (
        trace.wrap_forward(runner, lambda: 123, 1, "forward", NS(capturing=False, cudagraph_runtime_mode="NONE"))()
        == 123
    )
    assert [r["event"] for r in events(tmp_path)] == ["forward.begin", "forward.end"]


def test_disabled_factory_never_inspects_runtime(monkeypatch):
    package = ModuleType("vllm_ascend")
    package.envs = NS(VLLM_ASCEND_KV_TRACE="")
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    assert trace_module.KVTrace.from_env("worker", None) is None


def test_bound_observer_compares_after_full_forward_update(tmp_path):
    trace = writer(tmp_path, snapshots=True)
    runner, cache = make_runner()

    def complete_forward(num_tokens, *, _kv_trace_phase):
        assert num_tokens == 1 and _kv_trace_phase == "dummy"
        cache[1, 3] = 1  # Model execution.
        cache[1, 4] = 2  # Required post-model graph update.
        return 42

    runner.forward = trace.observe_forward(
        runner, complete_forward, lambda: NS(capturing=False, cudagraph_runtime_mode="FULL")
    )
    assert runner.forward(1, _kv_trace_phase="dummy") == 42
    diff = next(r for r in events(tmp_path) if r["event"] == "cache.diff")
    assert diff["changed_offsets"] == [3, 4]


class FakeManager:
    def __init__(self):
        self.mapping = {}
        self.block_pool = NS(blocks=[NS(ref_cnt=0), NS(ref_cnt=0)])
        self.kv_cache_config = NS(kv_cache_groups=[NS(layer_names=["layer"], kv_cache_spec=NS(block_size=8))])

    def get_block_ids(self, rid):
        return self.mapping.get(rid, ([],))

    def allocate_slots(self, request, num_new_tokens, fail=False):
        if fail:
            if request.request_id in self.mapping:
                self.mapping[request.request_id] = ([0],)
                self.block_pool.blocks[1].ref_cnt -= 1
            return None
        self.mapping[request.request_id] = ([1],)
        self.block_pool.blocks[1].ref_cnt += 1
        return self.mapping[request.request_id]

    def free(self, request):
        self.mapping.pop(request.request_id)
        self.block_pool.blocks[1].ref_cnt -= 1

    def cache_blocks(self, request, num_computed_tokens):
        return None

    def get_computed_blocks(self, request):
        return NS(get_block_ids=lambda: ([1],)), 8


def test_shared_blocks_release_reuse_and_allocation_failure(tmp_path):
    trace = writer(tmp_path)
    manager = FakeManager()
    manager_module.attach_manager_trace(manager, trace)
    requests = [NS(request_id=rid, status="RUNNING", num_computed_tokens=8) for rid in ("a", "b", "c")]
    a, b, c = requests
    assert manager.get_computed_blocks(a)[1] == 8
    manager.allocate_slots(a, 1)
    manager.allocate_slots(b, 1)
    manager.free(a)
    manager.free(b)
    assert manager.allocate_slots(c, 1, fail=True) is None
    manager.allocate_slots(c, 1)
    manager.cache_blocks(c, 8)
    manager.free(c)
    rows = events(tmp_path)
    acquisitions = [r for r in rows if r["event"] == "block.acquire"]
    assert [r["lease"] for r in acquisitions] == [1, 1, 2]
    assert [r["shared"] for r in acquisitions] == [False, True, False]
    assert [r["remaining_owners"] for r in rows if r["event"] == "block.release"] == [["b"], [], []]
    assert any(r["event"] == "block.allocation_failed" for r in rows)
    assert not manager._ascend_kv_trace.requests


def test_reader_links_pd_ids_and_preserves_span_completion(tmp_path):
    trace = writer(tmp_path)
    trace.emit("transfer.complete", request_id="decode", remote_request_id="prefill", local_block_ids=[[1]])
    trace.emit(
        "forward.begin", request_ids=["decode"], span_id="s", groups=[{"group_id": 0, "block_size": 8, "slots": [11]}]
    )
    trace.emit("cache.diff", span_id="s", group_id=0, block_id=1, changed_elements=4)
    trace.emit("forward.end", span_id="s")
    rows = cli.select_events(events(tmp_path), request="prefill")
    assert len(rows) == 4
    assert len(cli.select_events(rows, block=1, group=0, engine="engine")) == 4
    assert not cli.select_events(rows, block=1, group=1)


def test_failed_allocation_can_release_sliding_window_blocks(tmp_path):
    trace = writer(tmp_path)
    manager = FakeManager()
    manager_module.attach_manager_trace(manager, trace)
    request = NS(request_id="sliding", status="RUNNING", num_computed_tokens=8)
    manager.allocate_slots(request, 1)
    manager.allocate_slots(request, 1, fail=True)
    assert manager._ascend_kv_trace.owners[(0, 1)] == set()
    assert any(r["event"] == "block.release" for r in events(tmp_path))


def test_block_filter_preserves_finished_notifications(tmp_path):
    trace = writer(tmp_path)
    trace.emit("transfer.complete", request_id="decode", remote_request_id="prefill", local_block_ids=[[1]])
    trace.emit("transfer.worker_finished", request_id="decode")
    trace.emit("transfer.worker_finished", request_id="unrelated")
    assert [r["event"] for r in cli.select_events(events(tmp_path), block=1)] == [
        "transfer.complete",
        "transfer.worker_finished",
    ]


@pytest.mark.parametrize("kernel_id,offset", [(2, 3), (3, 7)])
def test_kernel_blocks_are_normalized_to_allocator_blocks(tmp_path, kernel_id, offset):
    trace = writer(tmp_path, snapshots=True)
    runner, cache = make_runner(block_size=4)
    cache = torch.zeros(4, 4, 1, 4)
    runner._kv_trace_caches["layer0"] = (cache,)
    runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size = 8
    table = runner.input_batch.block_table.block_tables[0]
    table.slot_mapping.gpu[:] = kernel_id * 4 + 3
    table.get_device_tensor = lambda: torch.tensor([[kernel_id]])
    trace.wrap_forward(
        runner, lambda: cache[kernel_id, 3].fill_(5), 1, "dummy", NS(capturing=False, cudagraph_runtime_mode="FULL")
    )()
    rows = events(tmp_path)
    diff = next(r for r in rows if r["event"] == "cache.diff")
    assert (diff["block_id"], diff["kernel_block_id"], diff["changed_offsets"]) == (1, kernel_id, [offset])
    assert set(cli.block_keys(rows[0])) == {(0, 1)}


def test_packed_nz_snapshot_is_explicitly_skipped(tmp_path):
    trace = writer(tmp_path, snapshots=True)
    runner, _ = make_runner()
    runner.ascend_config = NS(enable_kv_nz=True)
    assert (
        trace.wrap_forward(runner, lambda: 42, 1, "dummy", NS(capturing=False, cudagraph_runtime_mode="FULL"))() == 42
    )
    assert any(r.get("reason") == "packed_nz_layout" for r in events(tmp_path))


def test_transfer_logs_shard_metadata_without_payload(tmp_path):
    trace = writer(tmp_path)
    trace.transfer(
        "transfer.complete",
        {
            "request_id": "local",
            "remote_request_id": "remote",
            "payload": "never recorded",
            "group_pulls": [
                NS(group_id=2, remote_tp_offset=1, num_group_pulls=2, prefill_pp_rank=0, is_group_transfer_end=True)
            ],
        },
    )
    row = events(tmp_path)[0]
    assert row["group_pulls"][0]["group_id"] == 2
    assert row["group_pulls"][0]["is_group_transfer_end"]
    assert "payload" not in row
