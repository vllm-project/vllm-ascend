# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reader contract tests: access context never becomes fabricated ownership."""

import copy
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("relations_reader", Path(__file__).parents[3] / "tools/kv_block_trace.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class Recording:
    def __init__(self):
        self.events = []
        self.sequences = Counter()
        self.clock = 100

    def emit(self, writer, event, **fields):
        self.clock += 10
        self.sequences[writer] += 1
        row = {
            "schema_version": 2,
            "run_id": "run",
            "host": "node",
            "host_boot_id": "boot",
            "engine_id": "D",
            "dp_rank": 0,
            "tp_rank": 0,
            "pp_rank": 0,
            "pid": 100 if writer == "scheduler" else 200,
            "process_instance_id": "boot:100:1" if writer == "scheduler" else "boot:200:1",
            "trace_id": writer,
            "sequence": str(self.sequences[writer]),
            "event_id": f"{writer}:{self.sequences[writer]}",
            "monotonic_ns": str(self.clock),
            "wall_time_ns": str(1000 + self.clock),
            "event": event,
            **fields,
        }
        self.events.append(row)
        return row

    def allocate(self, epoch="1", request="spring", pool="pool"):
        self.emit("scheduler", "block.alloc", pool_id=pool, block_id=1, alloc_epoch=epoch)
        self.emit(
            "scheduler", "block.acquire", pool_id=pool, block_id=1, alloc_epoch=epoch, group_id=0, request_id=request
        )

    def bind(self, **fields):
        return self.emit("worker", "schedule.received", pool_id="pool", scheduler_id="scheduler", **fields)

    def transfer(self, name="transfer.complete", **fields):
        return self.emit(
            "transfer",
            name,
            request_id="spring",
            remote_request_id="prefill",
            remote_engine_id="P",
            local_block_ids=[[1]],
            **fields,
        )

    def begin(self, span="dummy", **fields):
        return self.emit(
            "worker",
            "forward.begin",
            phase="dummy",
            span_id=span,
            request_ids=[],
            block_refs=[],
            batch_request_ids=[],
            groups=[{"group_id": 0, "block_size": 128, "kernel_block_size": 128, "slots": [142], "block_table": [[1]]}],
            **fields,
        )

    def end(self, span="dummy"):
        diff = self.emit(
            "worker",
            "cache.diff",
            phase="dummy",
            span_id=span,
            block_id=1,
            group_id=0,
            pool_id=None,
            alloc_epoch=None,
            changed_offsets=[14],
            changed_elements=512,
        )
        self.emit("worker", "forward.end", phase="dummy", span_id=span)
        return diff


def case4():
    log = Recording()
    log.allocate()
    log.bind()
    log.transfer("transfer.start")
    log.transfer()
    log.begin()
    log.end()
    return log


def diffs(rows):
    return [e for e in rows if e["event"] == "cache.diff"]


def test_request_and_epoch_queries_retain_case4_dummy_without_assigning_ownership():
    log = case4()
    saved = copy.deepcopy(log.events)
    for selected in (
        reader.select_events(log.events, request="prefill"),
        reader.select_events(log.events, pool_id="pool", block=1, epoch="1"),
    ):
        diff = diffs(selected)[0]
        assert diff["pool_id"] is None and diff["alloc_epoch"] is None
        relation = diff["access_relations"][0]
        assert relation["owner_request_ids"] == ["spring"]
        assert relation["alloc_epoch"] == "1" and relation["pool_id"] == "pool"
        assert not relation["device_epoch_verified"]
        assert relation["latest_transfer"]["event"] == "transfer.complete"
        assert relation["latest_transfer"]["matches_recorded_owner"]
        assert not relation["context_changed_during_span"]
        assert any(e["event"] == "forward.end" for e in selected)
        event_ids = {e["event_id"] for e in selected}
        assert set(relation["evidence_event_ids"]).issubset(event_ids)
        assert relation["latest_transfer"]["event_id"] in event_ids
    assert log.events == saved
    assert next(e for e in reader.select_events(log.events) if e["event"] == "forward.begin")["request_ids"] == []


def test_reuse_does_not_attach_old_transfer_or_request_to_new_epoch():
    log = case4()
    log.emit("scheduler", "block.release", pool_id="pool", block_id=1, alloc_epoch="1", group_id=0, request_id="spring")
    log.allocate("2", "next")
    log.begin("next-dummy")
    log.end("next-dummy")
    old = diffs(reader.select_events(log.events, request="spring"))
    new = diffs(reader.select_events(log.events, pool_id="pool", block=1, epoch="2"))
    assert [e["span_id"] for e in old] == ["dummy"]
    assert [e["span_id"] for e in new] == ["next-dummy"]
    assert "latest_transfer" not in new[0]["access_relations"][0]


def test_shared_prefix_retains_multiple_readers_without_expanding_aliases():
    log = case4()
    log.emit("scheduler", "block.acquire", pool_id="pool", block_id=1, alloc_epoch="1", group_id=0, request_id="other")
    log.begin("shared")
    log.end("shared")
    log.emit("worker", "forward.begin", request_ids=["other"], span_id="unrelated-later", groups=[])
    selected = reader.select_events(log.events, request="spring")
    relation = next(e for e in diffs(selected) if e["span_id"] == "shared")["access_relations"][0]
    assert relation["owner_request_ids"] == ["other", "spring"]
    assert not any(e.get("span_id") == "unrelated-later" for e in selected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", "another-node"),
        ("host_boot_id", "new-boot"),
        ("dp_rank", 1),
        ("tp_rank", 1),
        ("pp_rank", 1),
        ("engine_id", "another-engine"),
        ("process_instance_id", "boot:200:2"),
    ],
)
def test_access_never_borrows_another_process_or_pool_scope(field, value):
    log = case4()
    for e in log.events:
        if e["event"].startswith("forward.") or e["event"] == "cache.diff":
            e[field] = value
    assert not diffs(reader.select_events(log.events, request="spring"))
    assert not diffs(reader.select_events(log.events, pool_id="pool", block=1, epoch="1"))


def test_foreign_transfer_is_not_latest_for_worker():
    log = case4()
    for e in log.events:
        if e["trace_id"] == "transfer":
            e["pid"] = 999
    relation = diffs(reader.select_events(log.events, request="spring"))[0]["access_relations"][0]
    assert "latest_transfer" not in relation


@pytest.mark.parametrize("writer", ["scheduler", "worker"])
def test_missing_sequence_or_explicit_gap_disables_inferred_epoch(writer):
    log = case4()
    if writer == "scheduler":
        log.events = [e for e in log.events if not (e["trace_id"] == writer and e["sequence"] == "1")]
    else:
        log.emit("worker", "trace.gap", reason="queue_overflow")
    assert not diffs(reader.select_events(log.events, pool_id="pool", block=1, epoch="1"))
    assert diffs(reader.select_events(log.events, block=1))  # Raw observation remains inspectable.


def test_out_of_order_wall_clock_does_not_change_local_relations():
    log = case4()
    for i, e in enumerate(log.events):
        e["wall_time_ns"] = str(9000 - i)
    relation = diffs(reader.select_events(list(reversed(log.events)), request="spring"))[0]["access_relations"][0]
    assert relation["alloc_epoch"] == "1"


def test_release_or_transfer_during_snapshot_is_visible_as_context_change():
    log = Recording()
    log.allocate()
    log.bind()
    log.transfer()
    log.begin()
    log.transfer("transfer.start")
    log.emit("scheduler", "block.release", pool_id="pool", block_id=1, alloc_epoch="1", group_id=0, request_id="spring")
    log.end()
    relation = diffs(reader.select_events(log.events, request="spring"))[0]["access_relations"][0]
    assert relation["context_changed_during_span"]
    assert relation["observed_at"] == "span_begin"
    selected = reader.select_events(log.events, request="spring")
    after = diffs(selected)[0]["access_relations"][1]
    assert after["observed_at"] == "span_end" and not after["owner_request_ids"]
    assert after["latest_transfer"]["event_id"] in {e["event_id"] for e in selected}


def test_reallocation_inside_snapshot_retains_both_boundary_contexts():
    log = Recording()
    log.allocate()
    log.bind()
    log.begin()
    log.emit("scheduler", "block.release", pool_id="pool", block_id=1, alloc_epoch="1", group_id=0, request_id="spring")
    log.allocate("2", "next")
    log.end()
    selected = reader.select_events(log.events, pool_id="pool", block=1, epoch="2")
    assert len(diffs(selected)) == 1
    assert [(r["alloc_epoch"], r["observed_at"]) for r in diffs(selected)[0]["access_relations"]] == [
        ("1", "span_begin"),
        ("2", "span_end"),
    ]


def test_unknown_epoch_is_not_filled_in_from_block_number():
    log = case4()
    for e in log.events:
        if e["event"] in ("block.alloc", "block.acquire"):
            e["alloc_epoch"] = None
    assert not diffs(reader.select_events(log.events, pool_id="pool", block=1, epoch="1"))


@pytest.mark.parametrize("scheduler_writer", ["scheduler", "zz-scheduler"])
def test_tied_host_timestamps_do_not_resolve_lifecycle_order(scheduler_writer):
    log = case4()
    begin = next(e for e in log.events if e["event"] == "forward.begin")
    for e in log.events:
        if e["event"] == "block.acquire":
            e["monotonic_ns"] = begin["monotonic_ns"]
        if e["trace_id"] == "scheduler":
            e["trace_id"] = scheduler_writer
        if e.get("scheduler_id") == "scheduler":
            e["scheduler_id"] = scheduler_writer
    selected = reader.select_events(log.events, request="spring")
    begin = next(e for e in selected if e["event"] == "forward.begin")
    assert begin["access_relations"][0]["association"] == "unknown"
    # A later boundary can be known without retrospectively ordering the tie.
    observations = diffs(selected)[0]["access_relations"]
    assert observations[0]["association"] == "unknown"
    assert observations[1]["observed_at"] == "span_end"
    assert observations[1]["owner_request_ids"] == ["spring"]


def test_unhealthy_transfer_writer_does_not_leave_a_stale_latest_transfer():
    log = case4()
    log.emit("transfer", "trace.gap", reason="missing_completion")
    relation = diffs(reader.select_events(log.events, request="spring"))[0]["access_relations"][0]
    assert "latest_transfer" not in relation


def test_group_and_padding_are_not_inferred_as_real_accesses():
    log = case4()
    begin = next(e for e in log.events if e["event"] == "forward.begin")
    begin["groups"][0]["slots"] = [-1, 0, 142]
    begin["groups"].append({"group_id": 1, "block_size": 128, "slots": [142], "block_table": [[1]]})
    context = next(e for e in reader.select_events(log.events) if e["event"] == "forward.begin")["access_relations"]
    assert [(r["group_id"], r["block_id"]) for r in context] == [(0, 1), (1, 1)]
    assert context[1]["owner_request_ids"] == []


def test_empty_owner_after_release_does_not_restore_previous_request():
    log = case4()
    log.emit("scheduler", "block.release", pool_id="pool", block_id=1, alloc_epoch="1", group_id=0, request_id="spring")
    log.begin("free-dummy")
    log.end("free-dummy")
    assert [e["span_id"] for e in diffs(reader.select_events(log.events, request="spring"))] == ["dummy"]
    free = next(
        e
        for e in diffs(reader.select_events(log.events, pool_id="pool", block=1, epoch="1"))
        if e["span_id"] == "free-dummy"
    )
    assert free["access_relations"][0]["owner_request_ids"] == []


def test_reloading_filtered_export_does_not_trust_old_derived_epoch():
    log = case4()
    exported_diff = diffs(reader.select_events(log.events, request="spring"))
    assert exported_diff[0]["access_relations"][0]["alloc_epoch"] == "1"
    assert not reader.select_events(exported_diff, pool_id="pool", block=1, epoch="1")


def test_legacy_log_uses_explicitly_labelled_host_pid_basis_and_rejects_ambiguity():
    log = case4()
    for e in log.events:
        e.pop("host_boot_id")
        e.pop("process_instance_id")
    relation = diffs(reader.select_events(log.events, request="spring"))[0]["access_relations"][0]
    assert relation["process_identity"] == "legacy_host_pid"
    log.emit("second-worker-writer", "schedule.received", pool_id="pool", scheduler_id="scheduler")
    log.events[-1].pop("host_boot_id")
    log.events[-1].pop("process_instance_id")
    assert not diffs(reader.select_events(log.events, request="spring"))


def test_cli_json_and_rank_filters(tmp_path):
    log = case4()
    file = tmp_path / "trace.jsonl"
    file.write_text("\n".join(json.dumps(e) for e in log.events), encoding="utf-8")
    command = [
        sys.executable,
        reader.__file__,
        str(file),
        "--request",
        "spring",
        "--json",
        "--host",
        "node",
        "--dp-rank",
        "0",
        "--tp-rank",
        "0",
        "--pp-rank",
        "0",
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(diffs(rows)) == 1
    assert not reader.select_events(log.events, request="spring", dp_rank=2)
