# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real transfer-range and snapshot code with a deterministic in-memory Store."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import (
    MEMCACHE_UNMATCHED_STATE,
    LayerwiseKVPoolWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta, RequestTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker


class MemoryStore:
    def __init__(self):
        self.addresses = {}
        self.committed = set()
        self.queries = []

    def batch_alloc(self, keys, sizes, ttl):
        for key in keys:
            self.addresses.setdefault(key, len(self.addresses) + 100)
        return [self.addresses[key] for key in keys]

    def batch_is_exist(self, keys):
        return [int(key in self.committed) for key in keys]

    def batch_get_key_info(self, keys, **kwargs):
        self.queries.extend(keys)
        return [
            SimpleNamespace(size=lambda: 4096, gva_list=lambda key=key: [self.addresses[key]])
            if key in self.committed
            else None
            for key in keys
        ]

    def batch_add_lease(self, keys, ttl):
        return [0 if key in self.committed else -1 for key in keys]


def make_worker():
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.use_layerwise = worker.use_layerwise_transfer = worker.layerwise_offload = True
    worker.use_block_key_layerwise = worker.block_key_hybrid = False
    worker.tp_size = 1
    worker.kv_role, worker.consumer_is_to_put = "kv_producer", False
    worker.can_put = True
    worker.num_kv_cache_groups, worker.num_layers = 1, 2
    worker.grouped_block_size, worker.hash_block_size = [128], 128
    worker.group_uses_align_state = [False]
    worker.cacheable_group_ids = {0}
    worker.independent_layers = []
    worker._reused_layers_by_group = {0: {0, 1}}
    worker._snapshot_keys = {}
    worker.cache_coordinator = None
    worker._is_layerwise_save_owner = lambda group: True
    worker._global_group_alloc_size = lambda group: 4096
    worker._make_layerwise_full_key = lambda group, key: "full:" + key
    worker._make_layerwise_partial_key = lambda request, group, block, end: f"{request.req_id}:{block}:{end}"
    worker._invalid_block_ids, worker._invalid_block_ids_lock = set(), threading.Lock()
    worker._save_disabled = threading.Event()
    worker.layerwise_max_transfer_blocks = 0
    worker.cache_transfer_granularity = 4096
    worker.m_store = MemoryStore()
    reset_step(worker)
    return worker


def reset_step(worker):
    worker._step_load_keys = []
    worker._masked_runs_cache = {}
    worker.layer_save_tasks, worker.layer_load_tasks = [[], []], [[], []]


def test_shared_prefix_releases_every_successful_read_lease():
    worker = make_worker()
    worker.m_store.addresses["shared"] = 100
    worker.m_store.committed.add("shared")
    worker.kv_recv_thread = None
    worker.m_store.batch_remove_lease = Mock(return_value=0)
    # Two requests may independently pin the same public object in one step.
    assert worker._lease_load_gvas(["shared"]) == [100]
    assert worker._lease_load_gvas(["shared"]) == [100]
    with patch.object(BaseKVPoolWorker, "get_finished", return_value=(set(), set())):
        worker.get_finished(set(), SimpleNamespace(preempted_req_ids=set()))
    worker.m_store.batch_remove_lease.assert_called_once_with(["shared", "shared"])
    assert not worker._step_load_keys


def test_batched_load_refreshes_shared_addresses_and_preserves_private_snapshots():
    worker = make_worker()
    for key in ("shared", "a-state", "b-state"):
        worker.m_store.addresses[key] = 100 + len(worker.m_store.addresses)
        worker.m_store.committed.add(key)

    def acquire(keys, ttl):
        # Emulate an address changing between the initial query and the lease.
        for key in keys:
            worker.m_store.addresses[key] += 1000
        return [0] * len(keys)

    worker.m_store.batch_add_lease = Mock(side_effect=acquire)

    def group(keys, partial=False):
        return np.zeros(len(keys), dtype=np.int64), list(range(len(keys))), keys, partial

    plans = [
        (SimpleNamespace(req_id="a"), [group(["shared"]), group(["a-state"], True)], {"shared": None}),
        (SimpleNamespace(req_id="b"), [group(["shared"]), group(["b-state"], True)], {"shared": None}),
        (SimpleNamespace(req_id="missing"), [group(["absent"])], {"absent": None}),
    ]
    addresses = list(worker._lease_step_load_gvas(plans))
    assert addresses == [[[1100], [1101]], [[1100], [1102]], None]
    assert worker._step_load_keys == ["shared", "a-state", "b-state"]
    assert [call.args[0] for call in worker.m_store.batch_add_lease.call_args_list] == [
        ["shared"],
        ["a-state"],
        ["b-state"],
    ]


def test_batched_private_partial_retry_keeps_the_owning_request():
    worker = make_worker()
    for key in ("a-full", "a-tail", "b-tail"):
        worker.m_store.addresses[key] = 100 + len(worker.m_store.addresses)
        worker.m_store.committed.add(key)
    worker.m_store.batch_add_lease = Mock(side_effect=[[0, MEMCACHE_UNMATCHED_STATE], [0], [0]])
    plans = [
        (SimpleNamespace(req_id="a"), [(np.zeros(2), [0, 1], ["a-full", "a-tail"], True)], {}),
        (SimpleNamespace(req_id="b"), [(np.zeros(1), [0], ["b-tail"], True)], {}),
    ]
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker.time.sleep"):
        assert list(worker._lease_step_load_gvas(plans)) == [[[100, 101]], [[102]]]
    assert [call.args[0] for call in worker.m_store.batch_add_lease.call_args_list] == [
        ["a-full", "a-tail"],
        ["a-tail"],
        ["b-tail"],
    ]
    assert worker._step_load_keys == ["a-full", "a-tail", "b-tail"]


@pytest.mark.parametrize("batch_limit", [1, 2])
def test_private_lease_batches_remap_partial_retry_and_release_once(batch_limit):
    worker = make_worker()
    worker.layerwise_max_transfer_blocks = batch_limit
    for key in ("a-0", "a-1", "a-2", "a-tail", "b-tail"):
        worker.m_store.addresses[key] = 100 + len(worker.m_store.addresses)
        worker.m_store.committed.add(key)
    tail_attempts = 0

    def acquire(keys, ttl):
        nonlocal tail_attempts
        results = [0] * len(keys)
        if "a-tail" in keys:
            tail_attempts += 1
            if tail_attempts == 1:
                results[keys.index("a-tail")] = MEMCACHE_UNMATCHED_STATE
        return results

    worker.m_store.batch_add_lease = Mock(side_effect=acquire)
    worker.m_store.batch_get_key_info = Mock(wraps=worker.m_store.batch_get_key_info)
    plans = [
        (SimpleNamespace(req_id="a"), [(np.zeros(4), [0, 1, 2, 3], ["a-0", "a-1", "a-2", "a-tail"], True)], {}),
        (SimpleNamespace(req_id="b"), [(np.zeros(1), [0], ["b-tail"], True)], {}),
    ]
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker.time.sleep"):
        assert list(worker._lease_step_load_gvas(plans)) == [[[100, 101, 102, 103]], [[104]]]
    assert tail_attempts == 2
    for method in (worker.m_store.batch_get_key_info, worker.m_store.batch_add_lease):
        assert all(len(call.args[0]) <= batch_limit for call in method.call_args_list)
    assert worker._step_load_keys == ["a-0", "a-1", "a-2", "a-tail", "b-tail"]
    worker.kv_recv_thread = None
    worker.m_store.batch_remove_lease = Mock(return_value=0)
    with patch.object(BaseKVPoolWorker, "get_finished", return_value=(set(), set())):
        worker.get_finished(set(), SimpleNamespace(preempted_req_ids=set()))
    worker.m_store.batch_remove_lease.assert_called_once_with(["a-0", "a-1", "a-2", "a-tail", "b-tail"])


@pytest.mark.parametrize("missing_info", [False, True])
def test_public_refresh_miss_isolated_and_successful_leases_released(missing_info):
    worker = make_worker()
    worker.m_store.addresses.update({"bad": 100, "good": 101})
    worker.m_store.committed.update(("bad", "good"))

    def acquire(keys, ttl):
        worker.m_store.addresses["good"] = 1101
        if missing_info:
            worker.m_store.committed.remove("bad")
        else:
            worker.m_store.addresses["bad"] = 0
        return [0] * len(keys)

    worker.m_store.batch_add_lease = Mock(side_effect=acquire)
    plans = [
        (SimpleNamespace(req_id=req), [(np.zeros(1), [0], [key], False)], {key: None})
        for req, key in (("bad-request", "bad"), ("good-request", "good"), ("shared-request", "good"))
    ]
    assert list(worker._lease_step_load_gvas(plans)) == [None, [[1101]], [[1101]]]
    assert worker._step_load_keys == ["bad", "good"]
    worker.m_store.batch_add_lease.assert_called_once()
    worker.kv_recv_thread = None
    worker.m_store.batch_remove_lease = Mock(return_value=0)
    with patch.object(BaseKVPoolWorker, "get_finished", return_value=(set(), set())):
        worker.get_finished(set(), SimpleNamespace(preempted_req_ids=set()))
    worker.m_store.batch_remove_lease.assert_called_once_with(["bad", "good"])


def request_meta(target, ids, *, cached=None, hashes=(), req_id="r"):
    return ReqMeta(
        req_id,
        target_token_len=target,
        save_end_token=target // 4096 * 4096,
        block_ids_by_group=[list(ids)],
        block_ids_by_group_np=[np.asarray(ids, dtype=np.int64)],
        block_hashes=list(hashes),
        can_save=True,
        load_spec=None if cached is None else LoadSpec(cached, cached, True, layerwise_previous_step=True),
        num_prompt_tokens=8192,
    )


def transferred(worker, *, save):
    builder = LayerBatchBuilder.__new__(LayerBatchBuilder)
    builder.group_id = 0
    builder._block_ids_buf = builder._block_gvas_buf = None
    tasks = worker.layer_save_tasks[1] if save else worker.layer_load_tasks[1]
    result = {}
    for task in tasks:
        data = builder.build_shared(task, is_save=save)
        result.update(zip(data.block_ids_arr.tolist(), data.block_gvas_arr.tolist()))
    return result


def save_request(worker, request):
    request.store_masks = worker._compute_reachable_store_masks(request)
    worker._process_save_for_layer_batch([request], 1)
    assert worker._alloc_gvas_for_save([request]) is not False
    data = transferred(worker, save=True)
    assert {worker.m_store.addresses[key] for key in request.save_keys} == set(data.values())
    worker.m_store.committed.update(request.save_keys)
    return data


@pytest.mark.parametrize("end", [127, 128, 129, 256, 385, 4095, 4096, 4097])
def test_exact_snapshot_survives_multiple_hash_boundaries(end):
    worker = make_worker()
    ids = list(range(1, (end + 127) // 128 + 2))
    saved = save_request(worker, request_meta(end, ids))
    assert set(saved) == set(ids[: (end + 127) // 128])
    reset_step(worker)
    meta = request_meta(end + 1, ids, cached=end, hashes=[f"h{i}" for i in range(end // 128)])
    assert worker._prepare_load_gvas_for_request([meta]) is not False
    worker._process_load_for_layer_batch([meta], 1)
    assert transferred(worker, save=False) == saved
    assert not worker._invalid_block_ids


def test_released_swa_ids_never_become_readable_keys():
    worker = make_worker()
    meta = request_meta(385, [0, 0, 9, 12])
    saved = save_request(worker, meta)
    assert set(saved) == {9, 12}
    assert set(worker._snapshot_keys["r"][0]) == {2, 3}
    reset_step(worker)
    continuation = request_meta(386, [0, 0, 9, 12], cached=385)
    assert worker._prepare_load_gvas_for_request([continuation]) is not False
    worker._process_load_for_layer_batch([continuation], 1)
    assert transferred(worker, save=False) == saved


def test_private_restore_does_not_apply_external_prefix_lookup_mask():
    worker = make_worker()
    worker.cache_coordinator = object()
    worker.token_database = SimpleNamespace(load_mask=Mock(return_value=([False, False, False, True],)))
    private = request_meta(386, [0, 0, 9, 12], cached=385)
    assert worker._compute_reachable_load_masks(private, 385) == ([False, False, True, True],)
    worker.token_database.load_mask.assert_not_called()
    private.load_spec.layerwise_previous_step = False
    assert worker._compute_reachable_load_masks(private, 385) == ([False, False, False, True],)
    worker.token_database.load_mask.assert_called_once()


def test_private_save_keeps_live_state_before_shared_slot_is_overwritten():
    worker = make_worker()
    worker.cache_coordinator = object()
    worker.token_database = SimpleNamespace(store_mask=Mock(return_value=([False] * 31 + [True],)))
    meta = request_meta(4096, [0] * 30 + [9, 12])
    assert worker._compute_reachable_store_masks(meta) == ([False] * 30 + [True, True],)
    meta.needs_private_snapshot = False
    assert worker._compute_reachable_store_masks(meta) == ([False] * 31 + [True],)


def test_glm_align_state_still_restores_only_the_exact_endpoint():
    worker = make_worker()
    worker.group_uses_align_state = [True]
    meta = request_meta(385, [0, 7, 8, 9], cached=384)
    assert worker._compute_reachable_load_masks(meta, 384) == ([False, False, True, False],)
    assert worker._compute_reachable_load_masks(meta, 383) == ([False, False, False, False],)


def test_request_private_snapshots_are_isolated():
    worker = make_worker()
    saved_a = save_request(worker, request_meta(7, [9], req_id="a"))
    reset_step(worker)
    saved_b = save_request(worker, request_meta(7, [12], req_id="b"))
    assert set(saved_a.values()).isdisjoint(saved_b.values())
    reset_step(worker)
    meta = request_meta(8, [9], cached=7, req_id="a")
    assert worker._prepare_load_gvas_for_request([meta]) is not False
    worker._process_load_for_layer_batch([meta], 1)
    assert transferred(worker, save=False) == saved_a


def test_missing_private_snapshot_fails_instead_of_loading_current_slot():
    worker = make_worker()
    meta = request_meta(8, [9], cached=7)
    assert worker._prepare_load_gvas_for_request([meta]) is False


def test_evicted_external_key_reports_a_failed_block():
    worker = make_worker()
    meta = request_meta(256, [9, 12], cached=128, hashes=["h0"])
    meta.load_spec.layerwise_previous_step = False
    assert worker._prepare_load_gvas_for_request([meta]) is False


def test_sliding_window_eviction_prunes_snapshot_identity():
    worker = make_worker()
    save_request(worker, request_meta(256, [5, 7]))
    assert set(worker._snapshot_keys["r"][0]) == {0, 1}
    reset_step(worker)
    save_request(worker, request_meta(257, [0, 7, 9]))
    assert set(worker._snapshot_keys["r"][0]) == {1, 2}


def test_null_partial_page_is_not_published_or_reloaded():
    worker = make_worker()
    assert save_request(worker, request_meta(7, [0])) == {}
    assert worker.m_store.addresses == {}
    assert worker._prepare_load_gvas_for_request([request_meta(8, [0], cached=7)]) is False


@pytest.mark.parametrize("use_eagle,expected", [(False, 32), (True, 30)])
def test_async_decode_snapshots_scheduled_tokens_before_ids_return(use_eagle, expected):
    scheduler = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
    scheduler.layerwise_offload, scheduler.use_eagle = True, use_eagle
    scheduler.cache_transfer_granularity = scheduler.hash_block_size = 128
    scheduler.original_block_size = [128]
    scheduler.kv_cache_group_families = ["attention"]
    scheduler._discard_partial_chunks = True
    request = SimpleNamespace(
        all_token_ids=list(range(30)),
        resumable=False,
        num_output_tokens=0,
        num_output_placeholders=2,
        max_tokens=16,
    )
    scheduler._unfinished_requests = {"r": (request, None)}
    tracker = RequestTracker("r", 32, allocated_block_ids_by_group=[[1]], num_prompt_tokens=30)
    meta = scheduler._build_req_meta(tracker, [], None, list(range(30)), False)
    assert meta.target_token_len == expected
    assert meta.can_save and meta.needs_private_snapshot
    # The next worker step must find the complete scheduled input extent.
    worker = make_worker()
    save_request(worker, meta)
    reset_step(worker)
    assert worker._prepare_load_gvas_for_request([request_meta(expected + 1, [1], cached=expected)]) is not False
