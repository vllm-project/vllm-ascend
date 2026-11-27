# SPDX-License-Identifier: Apache-2.0
"""Save references shared by full-page and partial-boundary jobs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import (
    InFlightSaveBlockTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendStoreKVConnectorWorkerMetadata,
    BoundaryPut,
    ReqMeta,
)


def _pool():
    return SimpleNamespace(blocks=list(range(20)), touch=Mock(), free_blocks=Mock())


def test_duplicate_boundary_sources_take_one_reference():
    tracker = InFlightSaveBlockTracker(2)
    pool = _pool()
    meta = ReqMeta(
        "r",
        token_len_chunk=48,
        block_ids_by_group=[[0, 1, 2], [3, 4]],
        can_save=True,
        boundary_puts=[BoundaryPut(0, 2, 48), BoundaryPut(1, 5, 48), BoundaryPut(0, 2, 48)],
    )
    tracker.retain(meta, pool)
    pool.touch.assert_called_once_with([2, 5])
    tracker.retain(meta, pool)
    assert pool.touch.call_count == 1
    tracker.complete({meta.event_id: 1}, pool)
    pool.free_blocks.assert_not_called()
    assert tracker.has_pending_work()
    tracker.complete({meta.event_id: 1}, pool)
    pool.free_blocks.assert_called_once_with([5, 2])
    assert not tracker.has_pending_work()


@pytest.mark.parametrize(
    "token_len_chunk, boundary_puts, expected_sources",
    [
        pytest.param(0, [BoundaryPut(0, 2, 16), BoundaryPut(1, 5, 16)], [2, 5], id="boundary-only"),
        pytest.param(16, [BoundaryPut(1, 5, 16)], [5], id="full-page-with-mamba-checkpoint"),
    ],
)
def test_save_pins_only_exact_boundary_sources(token_len_chunk, boundary_puts, expected_sources):
    tracker = InFlightSaveBlockTracker(1)
    pool = _pool()
    meta = ReqMeta(
        "r",
        token_len_chunk=token_len_chunk,
        can_save=True,
        block_ids_by_group=[[1, 2], [3, 4]],
        boundary_puts=boundary_puts,
    )
    tracker.retain(meta, pool)
    pool.touch.assert_called_once_with(expected_sources)
    tracker.complete({meta.event_id: 1}, pool)
    pool.free_blocks.assert_called_once_with(expected_sources[::-1])
    assert not tracker.has_pending_work()


def test_normal_save_without_boundary_puts_does_not_pin_ordinary_tables():
    tracker = InFlightSaveBlockTracker(1)
    pool = _pool()
    meta = ReqMeta("r", token_len_chunk=16, can_save=True, block_ids_by_group=[[0, 1, 2], [3, 4]])
    tracker.retain(meta, pool)
    pool.touch.assert_not_called()
    assert meta.event_id is None
    assert not tracker.has_pending_work()


def test_out_of_order_completion_does_not_release_another_job():
    tracker = InFlightSaveBlockTracker(1)
    pool = _pool()
    first = ReqMeta("r", can_save=True, boundary_puts=[BoundaryPut(0, 1, 16)])
    second = ReqMeta("r", can_save=True, boundary_puts=[BoundaryPut(0, 2, 32)])
    tracker.retain(first, pool)
    tracker.retain(second, pool)
    tracker.complete({second.event_id: 1}, pool)
    assert tracker.sending_blocks == {first.event_id: [1]}
    tracker.complete({first.event_id: 1}, pool)
    assert not tracker.has_tracked_blocks()


def test_aggregated_worker_counts_release_source():
    tracker = InFlightSaveBlockTracker(2)
    pool = _pool()
    meta = ReqMeta("r", can_save=True, boundary_puts=[BoundaryPut(0, 1, 16)])
    tracker.retain(meta, pool)
    first = AscendStoreKVConnectorWorkerMetadata({meta.event_id: 1})
    second = AscendStoreKVConnectorWorkerMetadata({meta.event_id: 1})

    completed = first.aggregate(second)

    assert completed.completed_events == {meta.event_id: 2}
    tracker.complete(completed.completed_events, pool)
    pool.free_blocks.assert_called_once_with([1])
    assert not tracker.has_pending_work()
