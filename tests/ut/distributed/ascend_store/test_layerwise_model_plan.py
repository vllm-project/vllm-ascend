# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise generation fences, transfer ownership and residency."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tests.ut.distributed.ascend_store.test_kv_transfer as transfer_tests
from tests.ut.distributed.ascend_store.test_layerwise_cache_layout import cache, names, runtime, spec
from vllm_ascend.core.layerwise_cache_config import get_layerwise_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import InFlightSaveBlockTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerLoadTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta


def receiver():
    thread, loads, saves, events = transfer_tests.TestGVALayerReceivingTaskOwnership()._make_thread()
    loads.extend([threading.Event(), threading.Event()])
    saves.extend([threading.Event(), threading.Event()])
    events.extend([MagicMock(), MagicMock()])
    thread.begin_generation(1)
    return thread, loads, saves, events


def test_multiple_predecessors_and_fanout_keep_completions():
    thread, _, saves, events = receiver()
    saves[0].set()
    saves[1].set()
    first = LayerLoadTask(0, [], 2, wait_for_save_layers=(1, 0), generation=1)
    assert thread._copy_gva_layers(first)
    second = LayerLoadTask(0, [], 3, generation=1)
    assert thread._copy_gva_layers(second)
    assert saves[0].is_set() and saves[1].is_set()
    assert events[0].synchronize.call_count == 2
    events[1].synchronize.assert_called_once()


def test_empty_load_cannot_overwrite_before_delayed_save():
    thread, _, saves, events = receiver()
    entered = threading.Event()
    thread.external_slot_release_waiter = lambda _: entered.set()
    result = []
    runner = threading.Thread(
        target=lambda: result.append(thread._copy_gva_layers(LayerLoadTask(0, [], 2, generation=1))), daemon=True
    )
    runner.start()
    assert not entered.wait(0.05)
    saves[0].set()
    runner.join(timeout=2)
    assert not runner.is_alive() and result == [True]
    assert entered.is_set()
    events[0].synchronize.assert_called_once()


def test_generation_rejects_stale_tasks_and_pending_queue():
    thread, _, _, _ = receiver()
    task = LayerLoadTask(None, [], 1, generation=1)
    thread.request_queue.put(task)
    with pytest.raises(RuntimeError, match="pending loads"):
        thread.begin_generation(2)
    thread.request_queue.get_nowait()
    thread.request_queue.task_done()
    thread.begin_generation(2)
    with pytest.raises(RuntimeError, match="Stale"):
        thread._copy_gva_layers(task)


def test_save_failure_never_releases_region():
    thread, _, saves, _ = receiver()
    saves[0].set()
    thread.save_failure_checker = MagicMock(side_effect=RuntimeError("compressed component failed"))
    thread.external_slot_release_waiter = MagicMock()
    with pytest.raises(RuntimeError, match="compressed component failed"):
        thread._copy_gva_layers(LayerLoadTask(0, [], 2, generation=1))
    thread.external_slot_release_waiter.assert_not_called()


@pytest.mark.parametrize("failure", ["copy", "publication"])
def test_backend_save_failure_wakes_waiters_but_rejects_reuse(failure):
    sender, store, finished, task = transfer_tests.TestGVALayerTransferFailures()._make_sending_thread()
    sender.require_complete_save = True
    if failure == "copy":
        store.store.batch_copy.return_value = -1
    else:
        store.batch_write_finish.return_value = [1]
    with pytest.raises(RuntimeError, match="Incomplete layerwise save"):
        sender._handle_request([task])
    assert finished.is_set() and sender.request_queue.unfinished_tasks == 0
    with pytest.raises(RuntimeError):
        sender.raise_if_failed()


def test_external_reader_must_release_before_overwrite():
    thread, _, saves, _ = receiver()
    saves[0].set()
    reading = threading.Event()
    released = threading.Event()

    def wait_for_reader(_):
        reading.set()
        assert released.wait(2)

    thread.external_slot_release_waiter = wait_for_reader
    finished = threading.Event()

    def restore():
        thread._copy_gva_layers(LayerLoadTask(0, [], 2, generation=1))
        finished.set()

    runner = threading.Thread(target=restore, daemon=True)
    runner.start()
    assert reading.wait(1) and not finished.is_set()
    released.set()
    runner.join(2)
    assert finished.is_set()


def test_cancelled_request_blocks_stay_pinned_until_all_ranks_complete():
    scheduler = object.__new__(LayerwiseKVPoolScheduler)
    tracker = scheduler._layerwise_save_blocks = InFlightSaveBlockTracker(2)
    scheduler.num_speculative_blocks_by_group = {1: 1}
    pool = scheduler._block_pool = MagicMock()
    pool.blocks = list(range(8))
    request = ReqMeta("cancelled", block_ids_by_group=[[0, 1, 2], [3, 4]], can_save=True)
    scheduler.retain_blocks_for_save(request)
    pool.touch.assert_called_once_with([1, 2, 3])
    assert scheduler.has_pending_push_work() and not scheduler.reset_store()
    # Cancellation releases the scheduler's own references, but each rank's
    # outstanding save still owns the retained references.
    request.can_save = False
    tracker.complete({request.event_id: 1}, pool)
    pool.free_blocks.assert_not_called()
    tracker.complete({request.event_id: 1}, pool)
    pool.free_blocks.assert_called_once_with([3, 2, 1])
    assert not scheduler.has_pending_push_work()


@pytest.mark.parametrize("cached_tokens", [0, 768])
def test_null_state_blocks_never_enter_model_load_masks(cached_tokens):
    worker = object.__new__(LayerwiseKVPoolWorker)
    worker._layerwise_layout = object()
    worker.cache_coordinator = None
    request = ReqMeta("r", block_ids_by_group=[[0, 2], [3, 0]])
    assert worker._compute_reachable_load_masks(request, cached_tokens) == ([False, True], [True, False])


def test_sparse_companion_transfers_with_its_rotating_group_layer():
    main = names(5)
    extra = names(5, suffix="extra")[4]
    worker = object.__new__(LayerwiseKVPoolWorker)
    worker.kv_cache_config = cache(dict.fromkeys([*main, extra], spec()), [main, [extra]])
    apply_layerwise_kv_cache_plan(worker.kv_cache_config, runtime())
    worker.hf_config = SimpleNamespace(num_hidden_layers=5)
    worker.num_layers = 5
    worker._layerwise_layout = get_layerwise_layout(worker.kv_cache_config)
    worker._global_to_local_layer = {i: i for i in range(5)}
    assert extra not in worker._layerwise_layout.rotating_layer_names
    assert not worker._is_resident_group_layer(4, 0)
