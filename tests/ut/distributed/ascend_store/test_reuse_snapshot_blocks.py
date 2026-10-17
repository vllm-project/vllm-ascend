# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker as module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta, RequestTracker


class Store:
    def __init__(self):
        self.data = {}
        self.committed = set()
        self.queries = []

    def batch_alloc(self, keys, sizes, ttl):
        for key in keys:
            self.data.setdefault(key, len(self.data) + 100)
        return [self.data[key] for key in keys]

    def batch_is_exist(self, keys):
        return [int(key in self.committed) for key in keys]

    def batch_get_key_info(self, keys, **kwargs):
        self.queries.extend(keys)
        assert all(key in self.committed for key in keys), keys
        return [SimpleNamespace(size=lambda: 4096, gva_list=lambda k=k: [self.data[k]]) for k in keys]

    def batch_add_lease(self, keys, ttl):
        assert all(key in self.committed for key in keys)
        return [0] * len(keys)


def worker():
    w = module.KVPoolWorker.__new__(module.KVPoolWorker)
    w.use_layerwise = w.use_layerwise_transfer = w.layerwise_offload = True
    w.use_block_key_layerwise = False
    w.tp_size = 16
    w.kv_role, w.consumer_is_to_put = "kv_producer", False
    w.can_put = True
    w.num_kv_cache_groups, w.num_layers = 1, 2
    w.grouped_block_size, w.hash_block_size = [128], 128
    w.group_uses_align_state = [False]
    w.cacheable_group_ids = {0}
    w.independent_layers = [0]
    w._is_layerwise_save_owner = lambda group: True
    w._global_group_alloc_size = lambda group: 4096
    w._compute_reachable_load_masks = lambda request, cached: None
    w._make_layerwise_full_key = lambda group, key: "full:" + key
    w._make_layerwise_partial_key = lambda r, group, block, end: f"{r.req_id}:snapshot:{block}:{end}"
    w._invalid_block_ids, w._invalid_block_ids_lock = set(), threading.Lock()
    w.layerwise_max_transfer_blocks = 0
    w.m_store = Store()
    reset(w)
    return w


def reset(w):
    w._masked_runs_cache = {}
    w._step_keyinfo_cache, w._step_lease_result, w._step_exist_cache = {}, {}, {}
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
            layerwise_snapshot_end_token=tracker.last_layerwise_snapshot_end_token,
            layerwise_snapshot_block_index=tracker.last_layerwise_snapshot_block_index,
            layerwise_snapshot_start_block=tracker.last_layerwise_snapshot_start_block,
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
    w._process_save_for_layer_batch([meta], 1)
    w._alloc_gvas_for_save([meta])
    addresses = transferred(w, True)
    # Every key being committed must have a transfer, not just an allocation.
    assert set(w.m_store.data[k] for k in meta.save_keys) == set(addresses.values())
    w.m_store.committed.update(meta.save_keys)
    return addresses


class SnapshotTests(unittest.TestCase):
    def test_crossing_then_hash_promotion(self):
        w = worker()
        t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1, 2, 3]])
        save(w, step(t, 127, []))
        reset(w)
        m = step(t, 129, [], 127)
        w._prepare_load_gvas([m])
        saved = save(w, m)
        self.assertEqual(set(saved), {0, 1})
        self.assertEqual(t.num_saved_tokens, 0)
        self.assertEqual(t.last_layerwise_snapshot_start_block, 0)
        reset(w)
        m = step(t, 131, ["h0"], 129)
        w._prepare_load_gvas([m])
        w._process_load_for_layer_batch([m], 1)
        self.assertEqual(transferred(w, False), saved)
        self.assertNotIn("full:h0", w.m_store.queries)
        save(w, m)
        self.assertIn("full:h0", w.m_store.committed)
        self.assertEqual(t.num_saved_tokens, 128)
        reset(w)
        m = step(t, 133, ["h0"], 131)
        w._prepare_load_gvas([m])
        self.assertIn("full:h0", w.m_store.queries)
        save(w, m)
        self.assertNotIn("full:h0", m.save_keys)

    def test_multiblock_snapshot_and_rollback(self):
        for end in (128, 129, 256, 385):
            for cached in (127, 128, end):
                if cached > end:
                    continue
                with self.subTest(end=end, cached=cached):
                    w = worker()
                    t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1, 2, 3]])
                    saved = save(w, step(t, end, []))
                    self.assertEqual(set(saved), set(range((end + 127) // 128)))
                    reset(w)
                    m = step(t, cached + 2, ["h" + str(i) for i in range(cached // 128)], cached)
                    w._prepare_load_gvas([m])
                    w._process_load_for_layer_batch([m], 1)
                    self.assertEqual(transferred(w, False), {i: saved[i] for i in range((cached + 127) // 128)})
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
                self.assertEqual(t.last_layerwise_snapshot_start_block, end // 128 if end % 128 else None)

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
        w._prepare_load_gvas([m])
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
            patch.object(module.time, "sleep") as sleep,
        ):
            w._prepare_load_gvas([m])
        query.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(w._invalid_block_ids, {0})
        self.assertFalse(m.load_keys)

    def test_partial_lease_retry_is_bounded(self):
        for recover in (False, True):
            with self.subTest(recover=recover):
                w = worker()
                t = RequestTracker("r", 0, allocated_block_ids_by_group=[[0, 1]])
                save(w, step(t, 127, []))
                reset(w)
                m = step(t, 129, [], 127)
                results = [[module.MEMCACHE_UNMATCHED_STATE], [0]] if recover else None
                with (
                    patch.object(
                        w.m_store,
                        "batch_add_lease",
                        return_value=[module.MEMCACHE_UNMATCHED_STATE],
                        side_effect=results,
                    ) as lease,
                    patch.object(module.time, "sleep") as sleep,
                ):
                    w._prepare_load_gvas([m])
                expected_retries = 1 if recover else 10
                self.assertEqual(lease.call_count, 1 + expected_retries)
                self.assertEqual(sleep.call_count, expected_retries)
                self.assertTrue(all(call.args == (0.001,) for call in sleep.call_args_list))
                self.assertEqual(w._invalid_block_ids, set() if recover else {0})
                self.assertEqual(bool(m.load_keys), recover)

    def test_other_data_planes_keep_their_save_range(self):
        legacy = LoadSpec(1, 1, True, None, 0, 7, 0, True)
        self.assertTrue(legacy.layerwise_previous_step)
        self.assertIsNone(legacy.layerwise_snapshot_start_block)
        w = worker()
        m = ReqMeta("r", target_token_len=385, save_end_token=128)
        w.use_layerwise_transfer = False
        self.assertEqual(w._layerwise_save_full_end(m, 0, 128, 3), 1)
        w.use_layerwise_transfer = True
        w.group_uses_align_state = [True]
        self.assertEqual(w._layerwise_save_full_end(m, 0, 128, 3), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
