# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure CPU diagnostics tests, runnable with --confcutdir=tests/ut/debug.

Load the dependency-free modules directly so the tests also run without a
complete vLLM/NPU installation (the project's parent conftest requires one).
"""

import importlib.util
import json
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    assert [int(r["sequence"]) for r in rows] == list(range(1, 82))
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


class AllocatorPool:
    """Small allocator with cached zero-reference blocks and a nonzero null ID."""

    def __init__(self):
        self.blocks = [NS(block_id=i, ref_cnt=0, block_hash=None) for i in range(3)]
        self.null_block = self.blocks[2]
        self.available = [0, 1]

    def get_new_blocks(self, num_blocks):
        if num_blocks > len(self.available):
            raise ValueError("exhausted")
        ids, self.available = self.available[:num_blocks], self.available[num_blocks:]
        for block_id in ids:
            self._maybe_evict_cached_block(self.blocks[block_id])
            self.blocks[block_id].ref_cnt += 1
        return [self.blocks[i] for i in ids]

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0:
                self.available.remove(block.block_id)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks, prepend=False):
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.available.append(block.block_id)

    def _maybe_evict_cached_block(self, block):
        had_hash = block.block_hash is not None
        block.block_hash = None
        return had_hash

    def reset_prefix_cache(self):
        if any(block.ref_cnt for block in self.blocks):
            return False
        for block in self.blocks:
            block.block_hash = None
        return True


class AllocatorManager(FakeManager):
    def __init__(self):
        super().__init__()
        self.block_pool = AllocatorPool()

    def allocate_slots(self, request, num_new_tokens, prefix=None):
        if prefix is None:
            blocks = self.block_pool.get_new_blocks(1)
        else:
            blocks = [self.block_pool.blocks[prefix]]
            self.block_pool.touch(blocks)
        self.mapping[request.request_id] = ([b.block_id for b in blocks],)
        return self.mapping[request.request_id]

    def free(self, request):
        ids = self.mapping.pop(request.request_id)[0]
        self.block_pool.free_blocks(self.block_pool.blocks[i] for i in ids)


@dataclass
class ScheduleMessage:
    num_scheduled_tokens: dict = field(default_factory=dict)
    finished_req_ids: set = field(default_factory=set)


@dataclass
class ExtendedScheduleMessage(ScheduleMessage):
    recomputed_reqs: list = field(default_factory=lambda: ["preserve-subclass-data"])


def test_epoch_tracks_allocation_not_zero_reference_prefix_reuse(tmp_path):
    trace = writer(tmp_path)
    manager = AllocatorManager()
    manager_module.attach_manager_trace(manager, trace)
    request = lambda rid: NS(request_id=rid, status="RUNNING", num_computed_tokens=8)
    a, b, c = map(request, ["a", "b", "c"])
    manager.allocate_slots(a, 1)  # block zero is real in this pool.
    manager.block_pool.blocks[0].block_hash = b"cached"
    manager.allocate_slots(b, 1, prefix=0)
    manager.free(a)
    manager.free(b)
    manager.allocate_slots(c, 1, prefix=0)
    manager.free(c)
    manager.block_pool.get_new_blocks(2)  # Physically repurpose both blocks.
    rows = events(tmp_path)
    acquired = [r for r in rows if r["event"] == "block.acquire"]
    assert [r["alloc_epoch"] for r in acquired] == ["1", "1", "1"]
    assert [r["lease"] for r in acquired] == [1, 1, 2]
    allocated = [r for r in rows if r["event"] == "block.alloc" and r["block_id"] == 0]
    assert [r["alloc_epoch"] for r in allocated] == ["1", "2"]
    assert any(r["event"] == "block.evict" and r["alloc_epoch"] == "1" for r in rows)
    assert not any(r.get("block_id") == 2 for r in rows)


def test_pool_observer_preserves_iterators_returns_errors_and_unknown_state(tmp_path):
    trace = writer(tmp_path)
    pool = AllocatorPool()
    observer = manager_module.attach_pool_trace(pool, trace)
    assert manager_module.attach_pool_trace(pool, trace) is observer
    blocks = pool.get_new_blocks(2)
    consumed = []

    def generate():
        for b in blocks:
            consumed.append(b.block_id)
            yield b

    pool.free_blocks(generate())
    assert consumed == [0, 1]
    assert pool.reset_prefix_cache() is True
    assert observer.epochs[:2] == [1, 1]
    with pytest.raises(ValueError, match="exhausted"):
        pool.get_new_blocks(3)
    assert observer.epochs == [None, None, None]
    pool.get_new_blocks(1)
    assert observer.epochs[0] == 2  # Do not repeat a generation after a gap.
    assert any(r["event"] == "trace.gap" for r in events(tmp_path))


def test_existing_cached_content_has_unknown_epoch(tmp_path):
    pool = AllocatorPool()
    pool.blocks[0].block_hash = b"predates-trace"
    observer = manager_module.attach_pool_trace(pool, writer(tmp_path))
    pool.touch([pool.blocks[0]])
    assert observer.reference(0)["alloc_epoch"] is None


def make_scheduler(trace):
    manager = AllocatorManager()
    manager_module.attach_manager_trace(manager, trace)
    request = NS(request_id="a", status="RUNNING", num_computed_tokens=0)
    manager.allocate_slots(request, 1)
    scheduler = NS(schedule=lambda: ExtendedScheduleMessage({"a": 1}), kv_cache_manager=manager)
    manager._ascend_kv_trace.wrap_schedule(scheduler)
    return scheduler, request


def test_schedule_pickle_worker_application_and_forward_share_context(tmp_path):
    scheduler_trace = writer(tmp_path, device_metadata=False)
    worker_trace = writer(tmp_path, device_metadata=False)
    for trace in (scheduler_trace, worker_trace):
        trace.emit("trace.manifest", implementation="p0-test")
    scheduler, _ = make_scheduler(scheduler_trace)
    output = scheduler.schedule()
    # Same protocol and out-of-band buffer route used by the pinned MessageQueue.
    buffers = []
    received = pickle.loads(
        pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffers.append), buffers=buffers
    )
    assert type(received) is ExtendedScheduleMessage
    assert received.recomputed_reqs == ["preserve-subclass-data"]
    context = received._ascend_kv_trace_context
    assert context == output._ascend_kv_trace_context
    runner = NS(requests={}, input_batch=NS(req_ids=["a"]), attn_state="DecodeOnly")
    deferred = lambda: "deferred-token-correction"

    def update(message):
        runner.requests["a"] = NS(block_ids=([0],))
        return deferred

    runner.update = worker_trace.observe_schedule(runner, update)

    def execute(message):
        assert runner.update(message) is deferred
        return worker_trace.wrap_forward(
            runner, lambda: 42, 1, "forward", NS(capturing=False, cudagraph_runtime_mode="FULL")
        )()

    runner.execute = worker_trace.observe_execution(runner, execute)
    assert runner.execute(received) == 42
    assert worker_trace._execution_context == {}
    rows = events(tmp_path)
    applied = next(r for r in rows if r["event"] == "schedule.apply")
    forward = next(r for r in rows if r["event"] == "forward.begin")
    assert applied["requests"][0]["mapping_matches"] is True
    assert applied["deferred_token_corrections"] is True
    assert forward["step_id"] == context["step_id"]
    assert forward["parent_event_ids"] == [applied["event_id"]]
    assert forward["block_refs"][0]["alloc_epoch"] == "1"
    assert forward["metadata_observation"] == "not_observed"
    assert forward["groups"] == []  # Runner has no device table at all.
    for trace in (scheduler_trace, worker_trace):
        trace.close()
    assert cli.check_events(events(tmp_path)) == []


def test_mapping_revision_changes_only_for_new_mapping_or_epoch(tmp_path):
    scheduler, request = make_scheduler(writer(tmp_path))
    first = scheduler.schedule()._ascend_kv_trace_context
    second = scheduler.schedule()._ascend_kv_trace_context
    assert first["step_id"] != second["step_id"]
    assert second["requests"][0]["mapping_revision"] == 1
    scheduler.kv_cache_manager.free(request)
    scheduler.kv_cache_manager.allocate_slots(request, 1)
    third = scheduler.schedule()._ascend_kv_trace_context
    assert third["requests"][0]["mapping_revision"] == 2


def test_dispatch_reads_upstream_new_request_req_id(tmp_path):
    trace = writer(tmp_path)
    manager = AllocatorManager()
    manager_module.attach_manager_trace(manager, trace)
    request = NS(request_id="new-request", status="RUNNING", num_computed_tokens=0)
    manager.allocate_slots(request, 1)
    message = ScheduleMessage({"new-request": 3})
    # vLLM NewRequestData uses req_id, whereas Request uses request_id.
    message.scheduled_new_reqs = [NS(req_id="new-request", num_computed_tokens=0)]
    scheduler = NS(schedule=lambda: message)
    manager._ascend_kv_trace.wrap_schedule(scheduler)
    output = scheduler.schedule()
    assert output._ascend_kv_trace_context["requests"][0]["num_computed_tokens"] == 0
    assert not any(row["event"] == "trace.observation_error" for row in events(tmp_path))


def test_worker_mismatch_missing_context_and_dummy_do_not_inherit_owners(tmp_path):
    trace = writer(tmp_path, device_metadata=False)
    scheduler, _ = make_scheduler(writer(tmp_path))
    output = scheduler.schedule()
    runner = NS(requests={"a": NS(block_ids=([1],))}, input_batch=NS(req_ids=["a"]), attn_state="DecodeOnly")
    update = trace.observe_schedule(runner, lambda message: None)

    def execute(message):
        update(message)
        return trace.wrap_forward(
            runner, lambda: "ok", 1, "dummy", NS(capturing=False, cudagraph_runtime_mode="FULL")
        )()

    observed = trace.observe_execution(runner, execute)
    assert observed(output) == "ok"
    assert observed(ScheduleMessage()) == "ok"
    trace.wrap_forward(runner, lambda: None, 1, "dummy", NS(capturing=False, cudagraph_runtime_mode="FULL"))()
    rows = events(tmp_path)
    assert any(r["event"] == "mapping.mismatch" for r in rows)
    assert any(r.get("reason") == "missing_or_incompatible_context" for r in rows)
    dummy = [r for r in rows if r["event"] == "forward.begin"]
    assert all(r["request_ids"] == [] and r["block_refs"] == [] for r in dummy)
    assert "step_id" not in dummy[-1]


def test_execution_exception_restores_context(tmp_path):
    trace = writer(tmp_path)

    def fail(message):
        raise RuntimeError("original model failure")

    with pytest.raises(RuntimeError, match="original model failure"):
        trace.observe_execution(NS(), fail)(ScheduleMessage())
    assert trace._execution_context == {}


def test_request_aliases_do_not_expand_through_batch_mates_or_other_runs(tmp_path):
    trace = writer(tmp_path)
    trace.transfer("transfer.complete", {"request_id": "decode", "remote_request_id": "prefill"})
    trace.emit("forward.begin", request_ids=["decode", "unrelated"], span_id="batch")
    trace.emit("forward.end", span_id="batch")
    trace.emit("forward.begin", request_ids=["unrelated"], span_id="later")
    trace.emit("forward.end", span_id="later")
    rows = events(tmp_path)
    selected = cli.select_events(rows, request="prefill")
    assert {e.get("span_id") for e in selected} == {None, "batch"}
    other = {**rows[-2], "run_id": "another-run", "request_ids": ["decode"]}
    assert other not in cli.select_events([*rows, other], request="prefill")


def test_pool_epoch_filter_does_not_mix_recycled_blocks(tmp_path):
    trace = writer(tmp_path)
    trace.emit("block.alloc", block_id=1, pool_id="one", alloc_epoch="1")
    trace.emit("block.alloc", block_id=1, pool_id="one", alloc_epoch="2")
    trace.emit("block.alloc", block_id=1, pool_id="two", alloc_epoch="1")
    selected = cli.select_events(events(tmp_path), block=1, pool_id="one", epoch="1")
    assert len(selected) == 1 and selected[0]["pool_id"] == "one" and selected[0]["alloc_epoch"] == "1"


def test_byte_budget_keeps_parseable_terminal_record_and_stops(tmp_path):
    trace = writer(tmp_path, max_trace_bytes=1)
    assert trace.emit("schedule.dispatch", run_id="kv-trace", requests=["x" * 4000]) is None
    trace.emit("ignored")
    rows = events(tmp_path)
    assert len(rows) == 1
    assert rows[0]["event"] == "trace.truncated"
    assert rows[0]["reason"] == "max_trace_bytes"
    assert rows[0]["run_id"] == "kv-trace"
    assert sum(p.stat().st_size for p in tmp_path.rglob("*.jsonl")) < 1024


def test_reader_detects_gaps_missing_parents_and_malformed_records(tmp_path):
    trace = writer(tmp_path)
    trace.emit("trace.manifest")
    trace.emit("event", parent_event_ids=["missing:7"])
    trace.close()
    file = next(tmp_path.rglob("*.jsonl"))
    lines = file.read_text().splitlines()
    file.write_text(lines[0] + "\n" + lines[2] + "\n{partial\n", encoding="utf-8")
    issues = []
    rows = cli.load_events([tmp_path, file], issues)
    assert len(rows) == 2  # Overlapping input paths do not duplicate records.
    assert issues[0]["reason"] == "invalid_or_incomplete_json"
    assert any(i["reason"] == "sequence_gap_or_duplicate" for i in cli.check_events(rows))
    parent_row = json.loads(lines[1])
    assert any(i["reason"] == "missing_parent" for i in cli.check_events([*rows, parent_row]))


def test_schema_v1_and_v2_sort_numeric_timestamps(tmp_path):
    rows = [
        dict(schema_version=v, run_id="r", trace_id=str(v), event="test", sequence=s, wall_time_ns=t)
        for v, s, t in [(1, 1, 10), (2, "1", "2")]
    ]
    file = tmp_path / "mixed.jsonl"
    file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert [int(r["wall_time_ns"]) for r in cli.load_events([file])] == [2, 10]


def test_cli_check_exit_status_and_epoch_argument_validation(tmp_path):
    trace = writer(tmp_path)
    trace.emit("trace.manifest")
    trace.close()
    command = [sys.executable, str(Path(cli.__file__)), str(tmp_path)]
    good = subprocess.run([*command, "--check"], capture_output=True, text=True)
    assert good.returncode == 0
    assert json.loads(good.stdout)["kv_integrity"] == "not_checked"
    invalid = subprocess.run([*command, "--epoch", "1"], capture_output=True, text=True)
    assert invalid.returncode == 2 and "requires --pool-id" in invalid.stderr
    file = next(tmp_path.rglob("*.jsonl"))
    file.write_text(file.read_text().splitlines()[0] + "\n", encoding="utf-8")
    incomplete = subprocess.run([*command, "--check"], capture_output=True, text=True)
    assert incomplete.returncode == 2


@pytest.mark.parametrize(
    "options", [{"device_metadata": "false"}, {"device_metadata": False, "snapshots": True}, {"max_trace_bytes": 0}]
)
def test_extended_config_validation(options):
    with pytest.raises(ValueError):
        trace_module.TraceConfig.parse(json.dumps({"directory": "/tmp/trace", **options}))


def test_scheduler_patch_instruments_bound_subclass_override_once(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug.kv_trace", trace_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.debug.kv_trace_manager", manager_module)
    patch = load_module("kv_patch_test_subject", "vllm_ascend/patch/platform/patch_kv_trace.py")
    trace = writer(tmp_path)
    monkeypatch.setattr(trace_module.KVTrace, "from_env", classmethod(lambda cls, *a, **k: trace))

    class BaseScheduler:
        def __init__(self):
            self.vllm_config = NS()
            self.kv_cache_manager = AllocatorManager()

        def schedule(self):
            raise AssertionError("Subclass override must be used")

    class CustomScheduler(BaseScheduler):
        def __init__(self):
            super().__init__()
            self.initialized = True

        def schedule(self):
            assert self.initialized
            return ExtendedScheduleMessage()

    patch.install_scheduler_trace(BaseScheduler)
    patch.install_scheduler_trace(BaseScheduler)
    scheduler = CustomScheduler()
    output = scheduler.schedule()
    assert type(output) is ExtendedScheduleMessage
    assert output._ascend_kv_trace_context["context_version"] == 1
    assert len([e for e in events(tmp_path) if e["event"] == "schedule.dispatch"]) == 1


def test_snapshot_uses_scheduler_null_block_identity(tmp_path):
    trace = writer(tmp_path, snapshots=True)
    runner, cache = make_runner()
    runner.input_batch.block_table.block_tables[0].slot_mapping.gpu[:] = 3  # Real block zero.
    trace._execution_context = {"null_block_id": 2}
    trace.wrap_forward(
        runner, lambda: cache[0, 3].fill_(8), 1, "dummy", NS(capturing=False, cudagraph_runtime_mode="FULL")
    )()
    diff = next(r for r in events(tmp_path) if r["event"] == "cache.diff")
    assert diff["block_id"] == 0 and diff["changed_offsets"] == [3]


def test_observation_failure_does_not_replace_model_result(tmp_path):
    trace = writer(tmp_path, device_metadata=False)
    scheduler, _ = make_scheduler(writer(tmp_path))
    runner = NS(requests=None, input_batch=NS(req_ids=[]), attn_state="DecodeOnly")
    update = trace.observe_schedule(runner, lambda output: None)

    def execute(output):
        update(output)
        return "model-result"

    assert trace.observe_execution(runner, execute)(scheduler.schedule()) == "model-result"
    assert any(r["event"] == "trace.observation_error" for r in events(tmp_path))
