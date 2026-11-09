# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model summary, generation fences and real strided transfer addresses."""

import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

import tests.ut.distributed.ascend_store.test_kv_transfer as transfer_tests
from tests.ut.distributed.ascend_store.test_layerwise_cache_layout import cache, names, runtime, spec
from tests.ut.distributed.ascend_store.test_segmented_cache_transfer import make_database, make_worker
from tests.ut.models.qwen4_exp.test_layerwise import PREFIX, make_config, make_views, runtime_config
from vllm_ascend.core.layerwise_cache_config import get_layerwise_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import InFlightSaveBlockTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_layerwise_policy,
    get_physical_layer_index,
    infer_layerwise_reuse_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerLoadTask, SharedBlockData
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta
from vllm_ascend.models.qwen4_exp.cache_config import build_qwen4_kv_cache_layout, make_qwen4_original_tensors


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


@pytest.mark.parametrize("consumer", ["scheduler", "worker"])
def test_consumers_reject_missing_published_plan(consumer):
    config = make_config(2)
    config.layerwise_reuse = None
    vc = runtime_config()
    if consumer == "scheduler":
        scheduler = object.__new__(LayerwiseKVPoolScheduler)
        scheduler.num_layers = 48
        with pytest.raises(ValueError, match="no published reuse layout"):
            scheduler._init_layerwise_offload(vc, config)
    else:
        worker = object.__new__(LayerwiseKVPoolWorker)
        worker.kv_cache_config = config
        worker.num_layers = 48
        worker.use_layerwise_transfer = True
        worker.hf_config = SimpleNamespace(num_hidden_layers=48)
        worker._extra_config = vc.kv_transfer_config.kv_connector_extra_config
        with pytest.raises(ValueError, match="no published reuse layout"):
            worker._init_layerwise_config()
    assert get_layerwise_layout(config) is None


def test_scheduler_and_worker_use_summary_and_owner_residency():
    config = make_config(2)
    original = build_qwen4_kv_cache_layout(config.kv_cache_groups, config.num_blocks)
    probe = KVCacheConfig(config.num_blocks, make_qwen4_original_tensors(original), config.kv_cache_groups)
    policy = get_layerwise_policy(probe, 48, runtime_config().kv_transfer_config.kv_connector_extra_config)
    _, expected = infer_layerwise_reuse_layout(probe, policy, base_layers=48)
    scheduler = object.__new__(LayerwiseKVPoolScheduler)
    scheduler.num_layers = 48
    scheduler._init_layerwise_offload(runtime_config(), config)
    assert scheduler.layerwise_offload
    worker = object.__new__(LayerwiseKVPoolWorker)
    worker.kv_cache_config = config
    worker.num_layers = 48
    worker.num_kv_cache_groups = 6
    worker.cacheable_group_ids = (1, 2, 3, 4, 5)
    worker.use_layerwise_transfer = True
    worker.use_layerwise = True
    worker.hf_config = SimpleNamespace(num_hidden_layers=48)
    worker._extra_config = runtime_config().kv_transfer_config.kv_connector_extra_config
    worker._init_layerwise_config()
    assert get_layerwise_layout(config) == scheduler._layerwise_layout == worker._layerwise_layout == expected
    assert worker.layerwise_offload
    assert worker._is_resident_group_layer(1, 5)
    assert worker._is_resident_group_layer(1, 2)
    assert not worker._is_resident_group_layer(2, 2)
    assert worker._reuse_predecessors[11] == (9,)
    assert worker._reuse_predecessors[5] == (3,)
    assert worker.num_layers == 48


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


@pytest.mark.parametrize(
    "suffix",
    [
        "3.self_attn.attn",
        "3.self_attn.indexer.compressed_key_cache",
        "2.linear_attn",
        "16.linear_attn",
        "17.linear_attn",
        "32.linear_attn",
        "33.linear_attn",
        "1.ple",
    ],
)
def test_model_views_use_real_pool_segments_for_selected_block(suffix):
    config = make_config(3)
    name = PREFIX + suffix
    group_id, group = next((i, group) for i, group in enumerate(config.kv_cache_groups) if name in group.layer_names)
    physical_layers = sorted({get_physical_layer_index(member, 48) for member in group.layer_names})
    layer_index = physical_layers.index(get_physical_layer_index(name, 48))

    def build():
        backings, all_caches = make_views(config)
        caches = all_caches[name]
        backing = backings[name] if name in backings else backings["shared"]
        backing.fill_(-1)
        worker = make_worker({name: caches}, layerwise=True)
        worker.kv_caches = all_caches
        worker._infer_cache_group_metadata(group_id, group.layer_names)
        database = make_database(worker, 384)
        database.block_size = [768] * len(config.kv_cache_groups)
        return backing, caches, database

    src, src_views, src_db = build()
    dst, dst_views, dst_db = build()
    expected, expected_views, _ = build()
    for index, tensor in enumerate(src_views):
        tensor.fill_(index + 2)
    shared = SharedBlockData(np.array([1]), np.array([4096]), ["request"], [False])

    def meta(db):
        builder = LayerBatchBuilder(db, sum(db.group_block_len[group_id]), 1, group_id=group_id)
        return builder.build_addrs(shared, layer_index)

    source, destination = meta(src_db), meta(dst_db)
    assert source.size_array.tolist() == destination.size_array.tolist()
    payload = [ctypes.string_at(int(addr), int(size)) for addr, size in zip(source.addr_array, source.size_array)]
    for addr, data in zip(destination.addr_array, payload):
        ctypes.memmove(int(addr), data, len(data))
    for source_view, expected_view in zip(src_views, expected_views):
        pages = source_view.shape[0] // 3
        expected_view[pages : 2 * pages].copy_(source_view[pages : 2 * pages])
    assert torch.equal(dst, expected)  # includes every guard, padding and unselected byte
