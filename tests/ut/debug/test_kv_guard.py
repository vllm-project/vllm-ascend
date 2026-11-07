# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 transitions, invalid accesses and exact Case4 address regression."""

import importlib.util
import sys
from pathlib import Path

import pytest

from . import complete_load, load_plan

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


def bind(guard, request="r", computed=0, epoch=1, pool="pool", ids=(1,)):
    guard.bind(
        {
            "pool_id": pool,
            "requests": {request: [[(block, epoch) for block in ids]]},
            "scheduled_requests": {request: (computed, 0)},
        }
    )
    if computed:
        complete_load(guard, request, computed)


def access(guard, slots=(), length=0, ids=(1,), request_ids=("r",), phase="forward"):
    guard.observe(
        [{"group_id": 0, "slots": list(slots), "block_table": [list(ids)], "seq_lens": [length]}],
        list(request_ids),
        phase,
    )


def violations(guard):
    return [e for e in guard.events if e["event"] == "guard.violation"]


def test_partial_fill_seal_and_read(guard):
    bind(guard)
    access(guard, range(128, 147), 19)
    assert guard.blocks[0, 1].valid_end == 19
    access(guard, range(147, 256), 128)
    assert guard.blocks[0, 1].valid_end == 128
    assert not violations(guard)


@pytest.mark.parametrize("reverse", [False, True])
def test_same_step_prefix_consumer_cannot_prevalidate_producers_write(guard, reverse):
    items = [("producer", [[(1, 1), (2, 1)]]), ("consumer", [[(1, 1), (3, 1)]])]
    guard.bind(
        {
            "pool_id": "pool",
            "requests": dict(reversed(items) if reverse else items),
            "scheduled_requests": {"producer": (0, 256), "consumer": (128, 128)},
        }
    )
    assert guard.blocks[0, 1].valid_end == 0
    guard.observe(
        [
            {
                "group_id": 0,
                "slots": list(range(128, 384)) + list(range(384, 512)),
                "block_table": [[1, 2], [1, 3]],
                "seq_lens": [256, 256],
            }
        ],
        ["producer", "consumer"],
        "forward",
    )
    assert not violations(guard)
    assert all(guard.blocks[0, block].valid_end == 128 for block in (1, 2, 3))


def test_computed_without_write_evidence_does_not_make_unknown_prefix_readable(guard):
    guard.bind({"requests": {"r": [[(1, 1)]]}, "scheduled_requests": {"r": (128, 0)}})
    access(guard, (), 128)
    assert violations(guard)[-1]["reason"] == "read_unwritten_range"


def test_recorded_prefix_survives_release_and_same_epoch_reuse(guard):
    bind(guard)
    access(guard, range(128, 256), 128)
    guard.bind({"released_request_ids": ["r"]})
    guard.bind({"requests": {"next": [[(1, 1)]]}, "scheduled_requests": {"next": (128, 0)}})
    access(guard, (), 128, request_ids=("next",))
    assert not violations(guard)
    access(guard, [128], 128, request_ids=("next",))
    assert violations(guard)[0]["reason"] == "overwrite_valid_prefix"


def test_unknown_read_and_skipped_write(guard):
    bind(guard)
    access(guard, (), 1)
    assert violations(guard)[-1]["reason"] == "read_unwritten_range"
    access(guard, [130])
    assert violations(guard)[-1]["reason"] == "noncontiguous_write"
    assert guard.blocks[0, 1].valid_end == subject.INVALID_PREFIX


def test_case4_loaded_prefix_dummy_write_then_dirty_read_without_tensors(guard):
    bind(guard)
    complete_load(guard)
    assert guard.blocks[0, 1].valid_end == 19
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
    bind(guard, request="new", epoch=2)
    assert guard.blocks[0, 1].valid_end == 0
    assert guard.blocks[0, 1].owners == {"new"}
    guard.events.clear()
    access(guard, [128], 1, request_ids=("new",))
    assert not violations(guard)


def test_shared_prefix_and_finished_request(guard):
    bind(guard, computed=128)
    bind(guard, request="shared", computed=128)
    access(guard, (), 128, request_ids=("shared",))
    assert not violations(guard)
    guard.bind({"released_request_ids": ["r"], "requests": {}})
    assert guard.blocks[0, 1].owners == {"shared"}
    access(guard, (), 128)
    assert violations(guard)[-1]["reason"] == "unowned_read"


def test_cache_lookup_does_not_revive_tainted_same_epoch(guard):
    bind(guard, computed=19)
    access(guard, [142], phase="dummy", request_ids=())
    guard.bind({"requests": {"prefix-hit": [[(1, 1)]]}, "scheduled_requests": {"prefix-hit": (19, 0)}})
    assert guard.blocks[0, 1].valid_end == subject.INVALID_PREFIX


def test_unregistered_completion_cannot_establish_validity(guard):
    bind(guard)
    guard.load(load_plan(guard))
    assert guard.blocks[0, 1].valid_end == 0
    access(guard, (), 19)
    assert violations(guard)[-1]["reason"] == "read_unwritten_range"
    assert any(e.get("reason") == "stale_load_completion" for e in guard.events)


def test_old_completion_never_applied_to_different_request(guard):
    bind(guard, request="old")
    meta = load_plan(guard, "old")
    guard.begin_load(meta)
    guard.bind({"released_request_ids": ["old"]})
    bind(guard, request="new", epoch=2)
    guard.load(meta)
    assert guard.blocks[0, 1].valid_end == 0
    assert not guard.loads


def test_no_per_token_state_log_and_negative_padding(guard):
    bind(guard)
    for offset in range(64):
        access(guard, [128 + offset, -1], offset + 1)
    assert not violations(guard)
    assert len([e for e in guard.events if e["event"] == "guard.state"]) == 0


def test_null_block_write_is_not_padding_and_taints_later_read(guard):
    access(guard, [-1, 0, 14], ids=(0,), request_ids=(), phase="dummy")
    assert violations(guard)[0]["reason"] == "write_null_block"
    assert violations(guard)[0]["offsets"] == [0, 14]
    access(guard, (), 1, ids=(0,))
    assert violations(guard)[-1]["reason"] == "read_null_block"
    guard.clear(0, 0)
    access(guard, (), 1, ids=(0,))
    assert violations(guard)[-1]["reason"] == "read_null_block"


def test_group_pool_and_allocation_isolation(guard):
    guard.configure({0: 128, 1: 128})
    bind(guard, computed=19)
    guard.bind({"pool_id": "pool", "requests": {"r2": [[], [(1, 1)]]}, "scheduled_requests": {"r2": (7, 0)}})
    complete_load(guard, "r2", 7)
    assert guard.blocks[0, 1].valid_end == 19
    assert guard.blocks[1, 1].valid_end == 7
    bind(guard, pool="new-pool")
    assert guard.blocks[0, 1].valid_end == 0
    assert (1, 1) not in guard.blocks


def test_copy_clear_and_read_invalid_destination(guard):
    bind(guard, computed=19)
    bind(guard, request="dest", ids=(2,))
    guard.copy_blocks(0, [(1, 2)])
    assert guard.blocks[0, 2].valid_end == 19
    access(guard, (), 19, ids=(2,), request_ids=("dest",))
    assert not violations(guard)
    guard.clear(0, 1)
    guard.copy_blocks(0, [(1, 2)])
    assert violations(guard)[-1]["reason"] == "invalid_copy_source_or_destination"
    access(guard, (), 19, ids=(2,), request_ids=("dest",))
    assert violations(guard)[-1]["reason"] == "read_invalid_block"


def test_recycling_a_live_block_is_a_lifecycle_violation(guard):
    bind(guard, computed=19)
    bind(guard, request="new", epoch=2)
    assert violations(guard)[0]["reason"] == "recycle_live_block"


def test_preempted_request_release_allows_reallocation_without_false_violation(guard):
    bind(guard, computed=19)
    guard.bind({"released_request_ids": ["r"]})
    access(guard, (), 19)
    assert violations(guard)[-1]["reason"] == "unowned_read"
    guard.events.clear()
    bind(guard, request="new", epoch=2)
    access(guard, [128], 1, request_ids=("new",))
    assert not violations(guard)


def test_block_copy_uses_original_sources_for_overlapping_pairs(guard):
    bind(guard, computed=19)
    bind(guard, request="b", computed=7, ids=(2,))
    bind(guard, request="c", ids=(3,))
    guard.copy_blocks(0, [(1, 2), (2, 3)])
    assert guard.blocks[0, 2].valid_end == 19
    assert guard.blocks[0, 3].valid_end == 7
    assert not violations(guard)


def test_missing_allocation_generation_does_not_make_a_block_writable(guard):
    bind(guard, epoch=0)
    assert (0, 1) not in guard.blocks
    assert any(e.get("reason") == "allocation_generation_missing" for e in guard.events)
    access(guard, [128])
    assert violations(guard)[-1]["reason"] == "unallocated_block"


def test_copying_an_empty_prefix_does_not_make_bytes_readable(guard):
    bind(guard)
    bind(guard, request="dest", ids=(2,))
    guard.copy_blocks(0, [(1, 2)])
    assert not violations(guard)
    access(guard, (), 1, ids=(2,), request_ids=("dest",))
    assert violations(guard)[-1]["reason"] == "read_unwritten_range"


def test_replaced_pool_drops_old_loads(guard):
    bind(guard)
    meta = load_plan(guard)
    guard.begin_load(meta)
    bind(guard, request="next", pool="replacement")
    guard.load(meta)
    assert not guard.loads
    assert guard.blocks[0, 1].valid_end == 0
