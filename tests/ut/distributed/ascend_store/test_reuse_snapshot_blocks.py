# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker as module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.gva import (
    MEMCACHE_UNMATCHED_STATE,
    GvaDataPlane,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta, RequestTracker


class Store:
    def __init__(self):
        self.data = {}
        self.committed = set()
        self.queries = []
        self.missing_info = None

    def batch_alloc(self, keys, sizes, ttl):
        for key in keys:
            self.data.setdefault(key, len(self.data) + 100)
        return [self.data[key] for key in keys]

    def batch_is_exist(self, keys):
        return [int(key in self.committed) for key in keys]

    def batch_get_key_info(self, keys, **kwargs):
        self.queries.extend(keys)
        return [
            SimpleNamespace(size=lambda: 4096, gva_list=lambda k=k: [self.data[k]])
            if k in self.committed
            else self.missing_info
            for k in keys
        ]

    def batch_add_lease(self, keys, ttl):
        assert all(key in self.committed for key in keys)
        return [0] * len(keys)


def worker():
    w = module.LayerwiseKVPoolWorker.__new__(module.LayerwiseKVPoolWorker)
    w.use_layerwise = w.use_layerwise_transfer = w.layerwise_offload = True
    w.use_block_key_layerwise = False
    w.tp_size = 16
    w.kv_role, w.save_decode_cache = "kv_producer", False
    w.can_put = True
    w.num_kv_cache_groups, w.num_layers = 1, 2
    w.grouped_block_size, w.hash_block_size = [128], 128
    w.group_uses_align_state = [False]
    w.cacheable_group_ids = {0}
    w.cache_coordinator = None
    w.dcp_size = 1
    w.cache_transfer_granularity = 128
    w.token_database = SimpleNamespace(load_mask=lambda *_: None, store_mask=lambda *_: None)
    w.vllm_config = SimpleNamespace(speculative_config=None)
    w._reused_layers_by_group = {0: {1}}
    w._snapshot_keys = {}
    w._is_layerwise_save_owner = lambda group: True
    w._global_group_alloc_size = lambda group: 4096
    w._compute_reachable_load_masks = lambda request, cached: None
    w._make_layerwise_full_key = lambda group, key: "full:" + key
    w.layerwise_keys = SimpleNamespace(make_full_key=lambda group, key, *ranks: "full:" + key)
    w.metadata = [SimpleNamespace(head_or_tp_rank=0)]
    w.pp_rank = w.pcp_rank = w.dcp_rank = 0
    w._make_layerwise_partial_key = lambda r, group, block, end: f"{r.req_id}:snapshot:{block}:{end}"
    w._invalid_block_ids, w._invalid_block_ids_lock = set(), threading.Lock()
    w._save_disabled = threading.Event()
    w.layerwise_max_transfer_blocks = 0
    w.m_store = Store()
    w._data_plane = GvaDataPlane(w)
    reset(w)
    return w


def set_rotating_window(w, spec):
    w.cache_coordinator = SimpleNamespace(attention_groups=[(spec, [0], module.SlidingWindowManager)])
    w._reused_layers_by_group.setdefault(0, {1})


def reset(w):
    w._step_load_keys = set()
    w._masked_runs_cache = {}
    w.layer_save_tasks, w.layer_load_tasks = [[], []], [[], []]


def step(tracker, target, hashes, cached=None):
    spec = (
        None
        if cached is None
        else LoadSpec(
            vllm_cached_tokens=cached,
            kvpool_cached_tokens=cached,
            can_load=True,
            layerwise_previous_step=True,
        )
    )
    tracker.token_len = target
    return ReqMeta.from_request_tracker(tracker, 128, block_hashes=hashes, load_spec=spec, save_partial_block=True)


def transferred(w, is_save):
    builder = LayerBatchBuilder.__new__(LayerBatchBuilder)
    builder.group_id = 0
    builder._block_ids_buf = builder._block_gvas_buf = None
    tasks = w.layer_save_tasks[1] if is_save else w.layer_load_tasks[1]
    if not tasks:
        return {}
    data = builder.build_shared(tasks[0], is_save=is_save)
    return dict(zip(data.block_ids_arr.tolist(), data.block_gvas_arr.tolist()))


def save(w, meta):
    w._data_plane._alloc_gvas_for_save([meta])
    w._process_save_for_layer_batch([meta], 1)
    addresses = transferred(w, True)
    # Every key being committed must have a transfer, not just an allocation.
    assert set(w.m_store.data[k] for k in meta.save_keys) == set(addresses.values())
    w.m_store.committed.update(meta.save_keys)
    return addresses


@pytest.mark.parametrize(
    "previous_step,public_ready",
    [(True, False), (False, False), (True, True)],
    ids=["continuation-fallback", "external-prefix", "public-preferred"],
)
def test_window_private_snapshot_only_falls_back_on_continuation(previous_step, public_ready):
    w = worker()
    t = RequestTracker("r", 0, allocated_block_ids_by_group=[[11, 12]], num_prompt_tokens=130)
    save(w, step(t, 128, []))
    private_key = w._snapshot_keys[t.req_id][0][0]
    set_rotating_window(w, SimpleNamespace(block_size=128, num_retained_tokens=1))
    del w._compute_reachable_load_masks
    reset(w)
    w.m_store.missing_info = SimpleNamespace(size=lambda: 0)
    if public_ready:
        w.m_store.data["full:h0"] = 999
        w.m_store.committed.add("full:h0")
    m = step(t, 129, ["h0"], cached=128)
    m.load_spec.layerwise_previous_step = previous_step
    w._data_plane._prepare_load_gvas([m])

    assert m.load_spec.can_load == previous_step
    key = "full:h0" if public_ready else private_key
    assert w._step_load_keys == ({key} if previous_step else set())
    if previous_step:
        w._process_load_for_layer_batch([m], 1)
        assert transferred(w, False) == {11: w.m_store.data[key]}
    if public_ready or not previous_step:
        assert private_key not in w.m_store.queries


def test_mixed_public_and_private_pages_preserve_logical_positions():
    w = worker()
    t = RequestTracker("r", 0, allocated_block_ids_by_group=[[31, 9, 47, 15]])
    save(w, step(t, 384, ["h0"]))
    private_key = w._snapshot_keys[t.req_id][0][2]
    reset(w)
    w.m_store.missing_info = SimpleNamespace(size=lambda: 0)
    w._compute_reachable_load_masks = lambda *_: ([True, False, True],)
    m = step(t, 385, ["h0", "h1", "h2"], cached=384)
    w._data_plane._prepare_load_gvas([m])
    w._process_load_for_layer_batch([m], 1)
    assert transferred(w, False) == {31: w.m_store.data["full:h0"], 47: w.m_store.data[private_key]}
    assert w._step_load_keys == {"full:h0", private_key}


@pytest.mark.parametrize("owner", [True, False], ids=["writer", "replica"])
def test_window_continuation_uses_native_full_keys_and_private_unhashed_pages(owner):
    import torch
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    w = worker()
    num_hashes = 32
    window_pages = [30, 31, 32, 33]
    w._is_layerwise_save_owner = lambda group: owner
    w.grouped_block_size, w.hash_block_size = [2], 2
    set_rotating_window(
        w, SlidingWindowSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.bfloat16, sliding_window=9)
    )
    del w._compute_reachable_load_masks  # Exercise the worker's actual continuation path.
    public = np.zeros(34, dtype=bool)
    public[[3, 8]] = True
    ids = np.arange(1, 35)
    ids[[8, 28]] = 0
    req = ReqMeta(
        "r",
        num_prompt_tokens=72,
        block_hashes=[f"h{i}" for i in range(num_hashes)],
        target_token_len=68,
        save_end_token=32,
        can_save=True,
        block_ids_by_group=[ids.tolist()],
        block_ids_by_group_np=[ids],
    )
    w.cache_transfer_granularity = 2
    w.token_database = SimpleNamespace(store_mask=lambda *_: (public,), load_mask=lambda *_: (public,))
    req.store_masks = w._compute_reachable_store_masks(req)
    assert np.flatnonzero(req.store_masks[0]).tolist() == [3, *window_pages]
    w._data_plane.prepare_sessions([req])
    public_pages = {3, *(page for page in window_pages if page < num_hashes)}
    assert {key for key in req.save_keys if key.startswith("full:")} == (
        {f"full:h{page}" for page in public_pages} if owner else set()
    )
    w._process_save_for_layer_batch([req], 1)
    assert set(transferred(w, True)) == ({int(ids[page]) for page in [3, *window_pages]} if owner else set())
    assert set(w._snapshot_keys["r"][0]) == set(window_pages) - public_pages
    req.load_spec = LoadSpec(68, 68, True, layerwise_previous_step=True)
    assert np.flatnonzero(w._compute_reachable_load_masks(req, 68)[0]).tolist() == window_pages
    assert np.flatnonzero(public).tolist() == [3, 8]


def test_rotating_window_rejects_missing_endpoint():
    end = 3
    w = worker()
    set_rotating_window(w, SimpleNamespace(block_size=2, num_retained_tokens=2))
    del w._compute_reachable_load_masks
    req = ReqMeta(
        "r",
        block_ids_by_group=[[11, 0]],
        block_ids_by_group_np=[np.array([11, 0])],
        can_save=True,
        num_prompt_tokens=4,
        target_token_len=end,
        load_spec=LoadSpec(end, end, True, layerwise_previous_step=True),
    )
    with pytest.raises(ValueError, match="Window endpoint"):
        w._data_plane._prepare_load_gvas([req])
    req.load_spec = None  # Isolate the write check after exercising the read check.
    with pytest.raises(ValueError, match="Window endpoint"):
        w._compute_reachable_store_masks(req)
    assert not w.m_store.queries and not w.m_store.data


def test_coordinator_window_group_can_include_resident_and_rotating_members():
    w = worker()
    w.num_kv_cache_groups = 2
    w.grouped_block_size = [2, 2]
    w.hash_block_size = w.cache_transfer_granularity = 2
    set_rotating_window(w, SimpleNamespace(block_size=2, num_retained_tokens=2))
    w.cache_coordinator.attention_groups[0][1].append(1)
    public = ([True, False, False], [True, False, False])
    w.token_database = SimpleNamespace(store_mask=lambda *_: public, load_mask=lambda *_: public)
    del w._compute_reachable_load_masks
    req = ReqMeta(
        "mixed",
        num_prompt_tokens=8,
        target_token_len=6,
        save_end_token=2,
        block_ids_by_group=[[10, 11, 12], [20, 21, 22]],
        block_hashes=["a", "b", "c"],
        can_save=True,
        load_spec=LoadSpec(6, 6, True, layerwise_previous_step=True),
    )
    save = w._compute_reachable_store_masks(req)
    load = w._compute_reachable_load_masks(req, 6)
    assert np.flatnonzero(save[0]).tolist() == [0, 2]
    assert np.flatnonzero(load[0]).tolist() == [2]
    assert save[1] is load[1] is public[1]
    req.load_spec.layerwise_previous_step = False
    assert w._compute_reachable_load_masks(req, 6) == public

    req.load_spec.layerwise_previous_step = True
    w._reused_layers_by_group.clear()
    assert w._compute_reachable_store_masks(req) is public
    assert w._compute_reachable_load_masks(req, 6) is public


class SnapshotTests(unittest.TestCase):
    def test_crossing_then_hash_promotion(self):
        w = worker()
        t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1, 2, 3]])
        save(w, step(t, 127, []))
        reset(w)
        m = step(t, 129, [], 127)
        w._data_plane._prepare_load_gvas([m])
        saved = save(w, m)
        self.assertEqual(set(saved), {0, 1})
        self.assertEqual(t.num_saved_tokens, 0)
        self.assertEqual(set(w._snapshot_keys[t.req_id][0]), {0, 1})
        reset(w)
        m = step(t, 131, ["h0"], 129)
        w._data_plane._prepare_load_gvas([m])
        w._process_load_for_layer_batch([m], 1)
        self.assertEqual(transferred(w, False), saved)
        self.assertIn("full:h0", w.m_store.queries)
        self.assertIn(w._snapshot_keys[t.req_id][0][0], w.m_store.queries)
        save(w, m)
        self.assertIn("full:h0", w.m_store.committed)
        self.assertNotIn(0, w._snapshot_keys[t.req_id][0])
        self.assertEqual(t.num_saved_tokens, 128)
        reset(w)
        m = step(t, 133, ["h0"], 131)
        w._data_plane._prepare_load_gvas([m])
        self.assertIn("full:h0", w.m_store.queries)
        save(w, m)
        self.assertNotIn("full:h0", m.save_keys)

    def test_multiblock_snapshot_restores_the_previous_prefill_extent(self):
        for end in (128, 129, 256, 385):
            with self.subTest(end=end):
                w = worker()
                t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1, 2, 3]])
                saved = save(w, step(t, end, []))
                self.assertEqual(set(saved), set(range((end + 127) // 128)))
                reset(w)
                m = step(t, end + 2, ["h" + str(i) for i in range(end // 128)], end)
                w._data_plane._prepare_load_gvas([m])
                w._process_load_for_layer_batch([m], 1)
                self.assertEqual(transferred(w, False), saved)
                self.assertFalse(w._invalid_block_ids)

    def test_known_hash_boundaries_and_prefill(self):
        for end in (128, 129, 8192):
            with self.subTest(end=end):
                w = worker()
                t = RequestTracker("r", 0, allocated_block_ids_by_group=[list(range(80))])
                m = step(t, end, ["h" + str(i) for i in range(end // 128)])
                saved = save(w, m)
                self.assertEqual(len(saved), (end + 127) // 128)
                self.assertEqual(t.num_saved_tokens, end // 128 * 128)
                self.assertEqual(set(w._snapshot_keys[t.req_id][0]), {end // 128} if end % 128 else set())

    def test_external_hit_is_not_previous_snapshot(self):
        w = worker()
        t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1]])
        m = step(t, 129, ["h0"])
        m.load_spec = LoadSpec(128, 128, True)
        save(w, m)
        self.assertNotIn("full:h0", m.save_keys)

    def test_independent_layer_does_not_reload_local_prefix(self):
        w = worker()
        t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1]])
        save(w, step(t, 129, []))
        reset(w)
        m = step(t, 131, ["h0"], 129)
        w._data_plane._prepare_load_gvas([m])
        w._process_load_for_layer_batch([m], 0)
        self.assertFalse(w.layer_load_tasks[0])

    def test_missing_previous_snapshot_does_not_poll(self):
        w = worker()
        t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1]])
        save(w, step(t, 127, []))
        reset(w)
        m = step(t, 129, [], 127)
        with (
            patch.object(w.m_store, "batch_get_key_info", return_value=[None]) as query,
            patch.object(time, "sleep") as sleep,
        ):
            w._data_plane._prepare_load_gvas([m])
        query.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(w._invalid_block_ids, {1})
        self.assertFalse(m.can_save)
        self.assertFalse(m.load_keys)

    def test_partial_lease_retry_is_bounded(self):
        for recover in (False, True):
            with self.subTest(recover=recover):
                w = worker()
                t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1]])
                save(w, step(t, 127, []))
                reset(w)
                m = step(t, 129, [], 127)
                results = [[MEMCACHE_UNMATCHED_STATE], [0]] if recover else None
                with (
                    patch.object(
                        w.m_store,
                        "batch_add_lease",
                        return_value=[MEMCACHE_UNMATCHED_STATE],
                        side_effect=results,
                    ) as lease,
                    patch.object(time, "sleep") as sleep,
                ):
                    w._data_plane._prepare_load_gvas([m])
                expected_retries = 1 if recover else 10
                self.assertEqual(lease.call_count, 1 + expected_retries)
                self.assertEqual(sleep.call_count, expected_retries)
                self.assertTrue(all(call.args == (0.001,) for call in sleep.call_args_list))
                self.assertEqual(w._invalid_block_ids, set() if recover else {1})
                self.assertEqual(bool(m.load_keys), recover)

    def test_other_data_planes_keep_their_save_range(self):
        w = worker()
        m = ReqMeta("r", target_token_len=385, save_end_token=128)
        w.use_layerwise_transfer = False
        self.assertEqual(w._layerwise_save_full_end(m, 128, 3), 1)
        w.use_layerwise_transfer = True
        w.layerwise_offload = False
        self.assertEqual(w._layerwise_save_full_end(m, 128, 3), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
