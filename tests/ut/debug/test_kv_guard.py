# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 transitions, invalid accesses and exact Case4 address regression."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "kv_guard_subject", Path(__file__).parents[3] / "vllm_ascend/debug/kv_guard.py"
)
assert spec is not None and spec.loader is not None
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


@pytest.fixture
def guard():
    events = []
    guard = subject.KVGuard(lambda event, **fields: events.append({"event": event, **fields}))
    guard.configure({0: 128})
    guard.events = events
    return guard


def bind(guard, request="r", computed=0, epoch="1", pool="pool", ids=(1,)):
    refs = [
        {"pool_id": pool, "alloc_epoch": epoch, "group_id": 0, "block_id": block, "logical_block_index": i}
        for i, block in enumerate(ids)
    ]
    guard.bind({"requests": [{"request_id": request, "num_computed_tokens": computed, "block_refs": refs}]})


def access(guard, slots=(), length=0, ids=(1,), request_ids=("r",), phase="forward"):
    guard.observe(
        [{"group_id": 0, "block_size": 128, "slots": list(slots), "block_table": [list(ids)], "seq_lens": [length]}],
        list(request_ids),
        phase,
    )


def violations(guard):
    return [e for e in guard.events if e["event"] == "guard.violation"]


def test_partial_fill_seal_and_read(guard):
    bind(guard)
    access(guard, range(128, 147), 19)
    assert guard.blocks[0, 1].state == subject.BlockState.PARTIAL
    assert guard.blocks[0, 1].valid_end == 19
    access(guard, range(147, 256), 128)
    assert guard.blocks[0, 1].state == subject.BlockState.SEALED_FILL
    assert not violations(guard)


def test_unknown_read_and_skipped_write(guard):
    bind(guard)
    access(guard, (), 1)
    assert violations(guard)[-1]["reason"] == "read_unwritten_range"
    access(guard, [130])
    assert violations(guard)[-1]["reason"] == "noncontiguous_write"
    assert guard.blocks[0, 1].state == subject.BlockState.INVALID


def test_case4_loaded_prefix_dummy_write_then_dirty_read_without_tensors(guard):
    bind(guard)
    guard.load({"request_id": "r", "local_block_ids": [[1]], "num_computed_tokens": 19})
    assert guard.blocks[0, 1].state == subject.BlockState.SEALED_LOAD
    access(guard, [142], 1, request_ids=(), phase="dummy")
    dirty_write = violations(guard)[0]
    assert dirty_write["reason"] == "dummy_write_to_live_block"
    assert dirty_write["block_id"] == 1 and dirty_write["offsets"] == [14]
    assert dirty_write["valid_end"] == 19 and dirty_write["owner_request_ids"] == ["r"]
    access(guard, (), 19)
    assert violations(guard)[-1]["reason"] == "read_invalid_block"


def test_loaded_tail_allows_append_but_not_overwrite(guard):
    bind(guard, computed=19)
    access(guard, [147], 20)
    assert not violations(guard)
    access(guard, [142], 20)
    assert violations(guard)[0]["reason"] == "overwrite_valid_prefix"


def test_reallocation_epoch_clears_old_validity_and_taint(guard):
    bind(guard, computed=19)
    access(guard, [142], phase="dummy", request_ids=())
    bind(guard, request="new", epoch="2")
    assert guard.blocks[0, 1].state == subject.BlockState.UNKNOWN
    assert guard.blocks[0, 1].owners == {"new"}
    guard.events.clear()
    access(guard, [128], 1, request_ids=("new",))
    assert not violations(guard)


def test_shared_prefix_and_finished_request(guard):
    bind(guard, computed=128)
    bind(guard, request="shared", computed=128)
    access(guard, (), 128, request_ids=("shared",))
    assert not violations(guard)
    guard.bind({"finished_request_ids": ["r"], "requests": []})
    assert guard.blocks[0, 1].owners == {"shared"}
    access(guard, (), 128)
    assert violations(guard)[-1]["reason"] == "unowned_read"


def test_cache_lookup_does_not_revive_tainted_same_epoch(guard):
    bind(guard, computed=19)
    access(guard, [142], phase="dummy", request_ids=())
    bind(guard, request="prefix-hit", computed=19)
    assert guard.blocks[0, 1].state == subject.BlockState.INVALID


def test_load_before_scheduler_binding_and_failed_load(guard):
    guard.load({"request_id": "r", "local_block_ids": [[1]], "num_computed_tokens": 19})
    bind(guard)
    assert not guard.pending_loads
    assert guard.blocks[0, 1].valid_end == 19
    guard.fail_load("r")
    access(guard, (), 19)
    assert violations(guard)[-1]["reason"] == "read_invalid_block"


def test_pending_load_never_applied_to_different_request(guard):
    guard.load({"request_id": "old", "local_block_ids": [[1]], "num_computed_tokens": 19})
    bind(guard, request="new", epoch="2")
    assert guard.blocks[0, 1].state == subject.BlockState.UNKNOWN
    guard.bind({"finished_request_ids": ["old"]})
    assert not guard.pending_loads


def test_no_per_token_state_log_and_negative_padding(guard):
    bind(guard)
    for offset in range(64):
        access(guard, [128 + offset, -1], offset + 1)
    assert not violations(guard)
    assert len([e for e in guard.events if e["event"] == "guard.state"]) == 1


def test_null_block_write_is_not_padding_and_taints_later_read(guard):
    access(guard, [-1, 0, 14], ids=(0,), request_ids=(), phase="dummy")
    assert violations(guard)[0]["reason"] == "write_null_block"
    assert violations(guard)[0]["offsets"] == [0, 14]
    access(guard, (), 1, ids=(0,))
    assert violations(guard)[-1]["reason"] == "read_tainted_null_block"
    guard.clear(0, 0)
    assert not guard.tainted_nulls


def test_nonlinear_cache_checks_reserved_destinations_without_append_rules(guard):
    bind(guard, computed=19)
    guard.observe(
        [{"group_id": 0, "block_size": 128, "write_mode": "reserved_only", "slots": [142, 0, -1]}], ["r"], "forward"
    )
    assert [v["reason"] for v in violations(guard)] == ["write_null_block"]
    assert guard.blocks[0, 1].valid_end == 19


def test_group_pool_and_allocation_isolation(guard):
    guard.configure({0: 128, 1: 128})
    bind(guard, computed=19)
    other = {"pool_id": "pool", "alloc_epoch": "1", "group_id": 1, "block_id": 1, "logical_block_index": 0}
    guard.bind({"requests": [{"request_id": "r2", "block_refs": [other], "num_computed_tokens": 7}]})
    assert guard.blocks[0, 1].valid_end == 19
    assert guard.blocks[1, 1].valid_end == 7
    bind(guard, pool="new-pool")
    assert guard.blocks[0, 1].state == subject.BlockState.UNKNOWN
    assert guard.blocks[1, 1].valid_end == 7


def test_copy_clear_and_read_invalid_destination(guard):
    bind(guard, computed=19)
    bind(guard, request="dest", ids=(2,))
    guard.copy(0, 1, 2, "dest")
    assert guard.blocks[0, 2].state == subject.BlockState.SEALED_LOAD
    access(guard, (), 19, ids=(2,), request_ids=("dest",))
    assert not violations(guard)
    guard.clear(0, 1)
    guard.copy(0, 1, 2, "dest")
    assert violations(guard)[-1]["reason"] == "invalid_copy_source_or_destination"
    access(guard, (), 19, ids=(2,), request_ids=("dest",))
    assert violations(guard)[-1]["reason"] == "read_invalid_block"


def test_recycling_a_live_block_is_a_lifecycle_violation(guard):
    bind(guard, computed=19)
    bind(guard, request="new", epoch="2")
    assert violations(guard)[0]["reason"] == "recycle_live_block"
