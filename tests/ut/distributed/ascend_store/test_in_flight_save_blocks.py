# SPDX-License-Identifier: Apache-2.0
"""Save references shared by full-page and partial-boundary jobs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import (
    InFlightSaveBlockTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import BoundaryPut, ReqMeta


def _pool():
    return SimpleNamespace(blocks=list(range(20)), touch=Mock(), free_blocks=Mock())


def test_boundary_and_ordinary_source_take_one_reference():
    tracker = InFlightSaveBlockTracker(2)
    pool = _pool()
    meta = ReqMeta(
        "r",
        block_ids_by_group=[[0, 1, 2], [3, 4]],
        can_save=True,
        boundary_puts=[BoundaryPut(0, 2, 48), BoundaryPut(1, 5, 48)],
    )
    tracker.retain(meta, {1: 1}, pool, retain_all=True)
    pool.touch.assert_called_once_with([2, 5, 1])
    tracker.retain(meta, {1: 1}, pool, retain_all=True)
    assert pool.touch.call_count == 1
    tracker.complete({meta.event_id: 1}, pool, {meta.event_id: {0}})
    tracker.complete({meta.event_id: 1}, pool, {meta.event_id: {0}})
    pool.free_blocks.assert_not_called()
    assert tracker.has_pending_work()
    tracker.complete({meta.event_id: 1}, pool, {meta.event_id: {1}})
    pool.free_blocks.assert_called_once_with([1, 5, 2])
    assert not tracker.has_pending_work()


def test_out_of_order_completion_does_not_release_another_job():
    tracker = InFlightSaveBlockTracker(1)
    pool = _pool()
    first = ReqMeta("r", can_save=True, boundary_puts=[BoundaryPut(0, 1, 16)])
    second = ReqMeta("r", can_save=True, boundary_puts=[BoundaryPut(0, 2, 32)])
    tracker.retain(first, {}, pool)
    tracker.retain(second, {}, pool)
    tracker.complete({second.event_id: 1}, pool, {second.event_id: {0}})
    assert tracker.sending_blocks == {first.event_id: [1]}
    tracker.complete({first.event_id: 1}, pool, {first.event_id: {0}})
    assert not tracker.has_tracked_blocks()


def test_invalid_rank_keeps_source_pinned():
    tracker = InFlightSaveBlockTracker(2)
    pool = _pool()
    meta = ReqMeta("r", can_save=True, boundary_puts=[BoundaryPut(0, 1, 16)])
    tracker.retain(meta, {}, pool)
    with pytest.raises(ValueError, match="Invalid completion rank"):
        tracker.complete({meta.event_id: 1}, pool, {meta.event_id: {2}})
    pool.free_blocks.assert_not_called()
    assert tracker.has_pending_work()


def test_full_page_legacy_path_pins_only_recurrent_non_scratch_sources():
    tracker = InFlightSaveBlockTracker(1)
    pool = _pool()
    meta = ReqMeta("r", block_ids_by_group=[[1, 2], [3, 4]], can_save=True)
    tracker.retain(meta, {1: 1}, pool)
    pool.touch.assert_called_once_with([3])
    tracker.complete({meta.event_id: 1}, pool)
    pool.free_blocks.assert_called_once_with([3])
