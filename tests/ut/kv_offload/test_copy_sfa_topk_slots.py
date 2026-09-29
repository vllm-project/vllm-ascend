# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fused_copy_sfa top-k slot binding and tail geometry."""

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    CopySfaRequestStates,
    CopySfaTopkSlotAllocator,
    copy_sfa_pool_capacity,
    copy_sfa_prefill_dest_geometry,
    copy_sfa_tail_geometry,
)


def test_copy_sfa_pool_capacity_includes_padding_rows():
    assert copy_sfa_pool_capacity(8) == 10


def test_copy_sfa_tail_geometry_skips_aligned_prefix():
    assert copy_sfa_tail_geometry(10240, 128) == (0, 0)
    assert copy_sfa_tail_geometry(0, 128) == (0, 0)


def test_copy_sfa_tail_geometry_keeps_incomplete_last_block():
    assert copy_sfa_tail_geometry(10367, 128) == (127, 80)
    assert copy_sfa_tail_geometry(129, 128) == (1, 1)


def test_copy_sfa_prefill_dest_geometry_dense_for_short_prompts():
    # Whole prompt fits the hot region: dense full-row D2D, including the
    # 128-aligned case that the tail path would skip entirely.
    assert copy_sfa_prefill_dest_geometry(129, 128, 8192) == (True, 0, 0)
    assert copy_sfa_prefill_dest_geometry(4096, 128, 8192) == (True, 0, 0)
    assert copy_sfa_prefill_dest_geometry(8192, 128, 8192) == (True, 0, 0)  # boundary
    assert copy_sfa_prefill_dest_geometry(0, 128, 8192) == (True, 0, 0)


def test_copy_sfa_prefill_dest_geometry_tail_for_long_prompts():
    # Longer than the hot budget: keep the circular-tail-only prefetch.
    assert copy_sfa_prefill_dest_geometry(10367, 128, 8192) == (False, 127, 80)
    # Block-aligned long prompt still has no tail to prefetch.
    assert copy_sfa_prefill_dest_geometry(8320, 128, 8192) == (False, 0, 0)


def test_copy_sfa_slot_allocator_reuses_and_releases():
    allocator = CopySfaTopkSlotAllocator(2)
    first = allocator.bind("req-a")
    second = allocator.bind("req-b")
    assert {first, second} == {0, 1}
    assert allocator.bind("req-a") == first
    allocator.release("req-a")
    assert allocator.get("req-a") is None
    reused = allocator.bind("req-c")
    assert reused == first


def test_copy_sfa_slot_allocator_exhausts_capacity():
    allocator = CopySfaTopkSlotAllocator(1)
    allocator.bind("req-a")
    with pytest.raises(RuntimeError, match="exhausted"):
        allocator.bind("req-b")


def _prepare(states, req_ids, lengths=None, *, live=None, prebound=None, dummy=False, capacity=4):
    slots = np.zeros(capacity, dtype=np.int32)
    generations = np.zeros(capacity, dtype=np.int64)
    restore, fills = states.prepare(
        req_ids=req_ids,
        live_req_ids=req_ids if live is None else live,
        slots=slots,
        generations=generations,
        prebound_slots={} if prebound is None else prebound,
        computed_tokens=None if lengths is None else np.asarray(lengths),
        padded_reqs=capacity,
        block_size=128,
        hot_tokens=8192,
        dummy=dummy,
    )
    return slots, generations, restore, fills


def test_recycled_slot_does_not_inherit_previous_request_rollback():
    states = CopySfaRequestStates()
    old = _prepare(states, ["old"], [8320], prebound={"old": 0})
    new = _prepare(states, ["new"], [512], prebound={"new": 0})
    assert old[0][0] == new[0][0] == 0
    assert new[1][0] > old[1][0]
    assert new[2:] == (False, {})


def test_same_request_rollback_across_hot_boundary_still_refills():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a"], [8320], prebound={"a": 1})
    after = _prepare(states, ["a"], [512], prebound={"a": 1})
    assert after[1][0] == before[1][0]
    assert after[2:] == (True, {1: (0, 512)})


def test_same_request_long_rollback_restores_only_tail():
    states = CopySfaRequestStates()
    _prepare(states, ["a"], [8704], prebound={"a": 1})
    after = _prepare(states, ["a"], [8448], prebound={"a": 1})
    assert after[2:] == (True, {})


def test_request_history_follows_identity_through_batch_reordering():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a", "b"], [8320, 512], prebound={"a": 2, "b": 0})
    after = _prepare(states, ["b", "a"], [640, 8448], prebound={"a": 2, "b": 0})
    assert after[0][:2].tolist() == before[0][:2].tolist()[::-1]
    assert after[1][:2].tolist() == before[1][:2].tolist()[::-1]
    assert after[2:] == (False, {})


def test_unscheduled_pd_request_keeps_its_reservation_and_history():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a"], [8320], prebound={"a": 0})
    _prepare(states, ["b"], [256], prebound={"a": 0})
    resumed = _prepare(states, ["a"], [512], prebound={"a": 0})
    assert resumed[0][0] == before[0][0]
    assert resumed[1][0] == before[1][0]
    assert resumed[3] == {int(before[0][0]): (0, 512)}


def test_unscheduled_unreserved_request_releases_its_slot():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a"], [8320], capacity=1)
    after = _prepare(states, ["b"], [512], capacity=1)
    assert after[0][0] == before[0][0]
    assert after[1][0] > before[1][0]
    assert after[2:] == (False, {})


def test_removed_then_resubmitted_same_id_gets_fresh_history():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a"], [8320], prebound={"a": 0})
    states.remove_request("a")
    states.remove_request("a")
    after = _prepare(states, ["a"], [512], prebound={"a": 0})
    assert after[1][0] > before[1][0]
    assert after[2:] == (False, {})


def test_finished_pd_owner_is_ignored_until_connector_cleanup():
    states = CopySfaRequestStates()
    before = _prepare(states, ["old"], [8320], prebound={"old": 0})
    states.remove_request("old")
    # Both bindings can coexist until the worker's post-forward get_finished.
    after = _prepare(states, ["new"], [512], prebound={"old": 0, "new": 0})
    assert after[1][0] > before[1][0]
    assert after[2:] == (False, {})
    stable = _prepare(states, ["new"], [640], prebound={"old": 0, "new": 0})
    assert stable[1][0] == after[1][0]
    _prepare(states, ["new"], [768], prebound={"new": 0})
    # Once cleanup removes the old reservation, its ID can be allocated anew.
    again = _prepare(states, ["old", "new"], [256, 896], prebound={"old": 1, "new": 0})
    assert again[1][0] > stable[1][0]
    assert again[2:] == (False, {})


def test_finished_waiting_pd_reservation_does_not_block_replacement():
    states = CopySfaRequestStates()
    states.remove_request("waiting")
    after = _prepare(states, ["new"], [512], prebound={"waiting": 0, "new": 0})
    assert after[0][0] == 0
    assert after[2:] == (False, {})


def test_waiting_pd_reservation_is_not_allocated_to_another_request():
    states = CopySfaRequestStates()
    other = _prepare(states, ["other"], [128], prebound={"waiting": 0})
    assert other[0][0] != 0
    ready = _prepare(states, ["waiting", "other"], [512, 256], prebound={"waiting": 0})
    assert ready[0][:2].tolist() == [0, int(other[0][0])]
    assert ready[1][1] == other[1][0]


def test_dummy_batch_does_not_change_live_request_history():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a"], [8320], prebound={"a": 1})
    dummy = _prepare(states, ["capture"], [0], live=[], dummy=True)
    assert dummy[0].tolist() == [4, 5, 6, 7]
    assert dummy[1].tolist() == [-1] * 4
    assert dummy[2:] == (False, {})
    after = _prepare(states, ["a"], [512], prebound={"a": 1})
    assert after[1][0] == before[1][0]
    assert after[2:] == (True, {1: (0, 512)})


def test_changed_live_pd_binding_fails_without_mutating_history():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a"], [8320], prebound={"a": 0})
    with pytest.raises(RuntimeError, match="binding changed"):
        _prepare(states, ["a"], [512], prebound={"a": 1})
    after = _prepare(states, ["a"], [512], prebound={"a": 0})
    assert after[1][0] == before[1][0]
    assert after[2:] == (True, {0: (0, 512)})


def test_pd_reservation_cannot_steal_a_live_request_slot():
    states = CopySfaRequestStates()
    _prepare(states, ["a"], [8320])
    with pytest.raises(RuntimeError, match="still owned"):
        _prepare(states, ["b"], [512], live=["a", "b"], prebound={"b": 0})


@pytest.mark.parametrize("prebound", [{"a": 0, "b": 0}, {"a": -1}, {"a": 4}])
def test_invalid_pd_bindings_are_rejected(prebound):
    with pytest.raises((ValueError, RuntimeError), match="slot"):
        _prepare(CopySfaRequestStates(), ["a"], [512], prebound=prebound)


def test_duplicate_batch_request_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _prepare(CopySfaRequestStates(), ["a", "a"], [128, 128])


def test_pool_exhaustion_preserves_existing_request_state():
    states = CopySfaRequestStates()
    before = _prepare(states, ["a"], [8320], capacity=1)
    with pytest.raises(RuntimeError, match="exhausted"):
        _prepare(states, ["b"], [512], live=["a", "b"], capacity=1)
    after = _prepare(states, ["a"], [512], capacity=1)
    assert after[1][0] == before[1][0]
    assert after[3] == {0: (0, 512)}
