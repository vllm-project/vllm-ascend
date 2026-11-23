# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise shared-source scheduling through the real layerwise GVA data plane."""

import ctypes
import threading
import unittest
from collections import Counter
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from tests.ut.distributed.ascend_store import test_pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
    create_kv_pool_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, LoadSpec, ReqMeta


class MemoryGVAStore:
    """Copy real CPU tensor descriptors to simulated GVA objects."""

    requires_exists_before_put = True

    def __init__(self):
        self.store = self
        self.pool = bytearray(1 << 20)
        self.next_gva = 4096
        self.objects = {}
        self.published = set()
        self.leases = set()
        self.copies = []
        self.finish_calls = []
        self.block_address = None
        self.block_direction = 1
        self.blocked = threading.Event()
        self.resume = threading.Event()
        self.fail_copy = False

    def ensure_initialized(self):
        pass

    def set_device(self):
        pass

    def register_buffer(self, *_args):
        pass

    def close(self):
        pass

    def exists(self, keys):
        return [int(key in self.published) for key in keys]

    def batch_is_exist(self, keys):
        return self.exists(keys)

    def batch_is_readable(self, keys):
        return [key in self.published for key in keys]

    def batch_alloc(self, keys, sizes, _ttl=0):
        result = []
        for key, size in zip(keys, sizes, strict=True):
            if key not in self.objects:
                self.objects[key] = (self.next_gva, size)
                self.next_gva += size
            result.append(self.objects[key][0])
        return result

    def batch_copy(self, gvas, addrs, sizes, direction):
        if direction == self.block_direction and self.block_address in addrs:
            self.blocked.set()
            if not self.resume.wait(timeout=5):
                raise RuntimeError("test copy was not released")
        if self.fail_copy:
            return -1
        self.copies.append((direction, list(addrs)))
        for gva, addr, size in zip(gvas, addrs, sizes, strict=True):
            if direction == 0:
                self.pool[gva : gva + size] = ctypes.string_at(addr, size)
            else:
                ctypes.memmove(addr, bytes(self.pool[gva : gva + size]), size)
        return 0

    def batch_write_finish(self, keys, results):
        self.finish_calls.append((list(keys), list(results)))
        self.published.update(key for key, result in zip(keys, results, strict=True) if result == 0)
        return [0] * len(keys)

    def batch_get_key_info(self, keys, *, for_load=False):
        return [
            SimpleNamespace(gva_list=lambda k=key: [self.objects[k][0]], size=lambda k=key: self.objects[k][1])
            if key in self.published
            else None
            for key in keys
        ]

    def batch_add_lease(self, keys, _ttl=0):
        self.leases.update(key for key in keys if key in self.published)
        return [0 if key in self.published else -1 for key in keys]

    def batch_remove_lease(self, keys):
        self.leases.difference_update(keys)
        return 0


def make_shared_worker(test, *, bounded=False, draft=False, num_layers=12, sources=(2, 8)):
    """Register real CPU cache views with the production GVA worker and queue threads."""
    names = [
        name
        for source in sources
        for name in (f"model.layers.{source}.attn.long_kv_cache", f"model.layers.{source}.attn.indexer.k_cache")
    ]
    groups = [names[::2], names[1::2]]
    if not bounded:
        groups.append([f"model.layers.{layer}.attn.swa_cache" for layer in range(num_layers)])
    if draft:
        groups.append(["model.mtp.layers.0.attn.kv_cache"])
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.uint8)
    cache_config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(layer_names=group, kv_cache_spec=spec) for group in groups],
        dspark_draft_layer_names=set(groups[-1]) if draft else set(),
        num_blocks=4,
        prefix_cache_retention_interval=0,
    )

    def create(config, use_layerwise, kv_cache_config):
        config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
        config.model_config.hf_config = config.model_config.hf_text_config = SimpleNamespace(
            num_hidden_layers=num_layers, kv_source_layer_ids=list(sources), compress_ratios=[2] * num_layers
        )
        config.model_config.enforce_eager = True
        config.compilation_config.cudagraph_mode = SimpleNamespace(name="NONE")
        config.compilation_config.static_forward_context = {
            str(layer): SimpleNamespace(
                layer_idx=layer, kv_transfer_cache_names=(f"model.layers.{layer}.attn.swa_cache",)
            )
            for layer in range(num_layers)
        }
        config.parallel_config.use_ubatching = config.parallel_config.use_sequence_parallel_moe = False
        config.cache_config.swa_bounded_replay = bounded
        config.use_v2_model_runner = True
        config.speculative_config = None
        return create_kv_pool_worker(config, use_layerwise, kv_cache_config)

    with (
        patch.object(test_pool_worker, "create_kv_pool_worker", side_effect=create),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.uses_hybrid_kv_cache",
            return_value=True,
        ),
    ):
        worker = test_pool_worker.make_worker(
            test,
            use_layerwise=True,
            num_layers=num_layers,
            num_hidden_layers=num_layers,
            extra_config={"backend": "memcache"},
            kv_cache_config=cache_config,
        )
    store = MemoryGVAStore()
    worker.m_store = store
    worker.cache_coordinator = worker.token_database.cache_coordinator = None
    all_names = sum(groups, [])
    alignment = 2 * 1024 * 1024
    slot_bytes = 64
    backing = torch.empty(alignment + len(all_names) * slot_bytes, dtype=torch.uint8)
    start = -backing.data_ptr() % alignment
    caches = {
        name: backing[start + index * slot_bytes : start + index * slot_bytes + 64].reshape(4, 16, 1).fill_(index + 1)
        for index, name in enumerate(all_names)
    }

    def close():
        store.resume.set()
        with suppress(RuntimeError):
            worker.shutdown()

    test.addCleanup(close)
    worker.register_kv_caches(caches)
    return worker, store, caches


def metadata(worker, *, load=False, save=True):
    result = AscendConnectorMetadata(set())
    result.add_request(
        ReqMeta(
            "r",
            token_len_chunk=16,
            num_prompt_tokens=32,
            block_hashes=[b"h"],
            block_ids_by_group=[[1] for _ in worker.grouped_block_size],
            block_ids_np=np.asarray([1], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([1], dtype=np.int64) for _ in worker.grouped_block_size],
            kv_cache_group_ids=list(worker.cacheable_group_ids),
            can_save=save,
            load_spec=LoadSpec(0, 16, True) if load else None,
        )
    )
    return result


def forward_layer(worker, meta):
    worker.wait_for_layer_load()
    worker.save_kv_layer(meta)


def save_prefix(worker):
    meta = metadata(worker)
    worker.prepare_layerwise_step(meta)
    worker.start_load_kv(meta)
    for layer in range(worker.num_layers):
        forward_layer(worker, meta)
    worker.get_finished(set(), meta)


class TestSharedSourceLayerwise(unittest.TestCase):
    def test_sparse_groups_keep_model_layer_slots(self):
        worker, store, caches = make_shared_worker(self, bounded=True)
        self.assertIs(type(worker), LayerwiseKVPoolWorker)
        self.assertEqual(worker.num_layers, 12)
        self.assertFalse(worker.layerwise_offload)
        self.assertEqual(worker._groups_for_layerwise_transfer(2), [(0, 0), (1, 0)])
        self.assertEqual(worker._groups_for_layerwise_transfer(8), [(0, 1), (1, 1)])
        self.assertEqual(worker._groups_for_layerwise_transfer(0), [])
        save_prefix(worker)
        self.assertTrue(store.published)
        self.assertEqual({size for _, size in store.objects.values()}, {32})

    def test_roundtrip_sparse_sources_swa_and_draft(self):
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                worker, store, caches = make_shared_worker(self, bounded=bounded, draft=True)
                expected = {name: cache.clone() for name, cache in caches.items()}
                save_prefix(worker)
                for cache in caches.values():
                    cache.zero_()
                meta = metadata(worker, load=True, save=False)
                worker.prepare_layerwise_step(meta)
                worker.start_load_kv(meta)
                for layer in range(worker.num_layers):
                    forward_layer(worker, meta)
                worker.get_finished(set(), meta)
                for name, cache in caches.items():
                    torch.testing.assert_close(cache[1], expected[name][1])
                    self.assertEqual(int(cache[0].sum()), 0)
                self.assertFalse(store.leases)

    def test_source_triggered_submission_preserves_swa_order_and_tail(self):
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                worker, store, _ = make_shared_worker(
                    self, bounded=bounded, draft=True, num_layers=24, sources=(2, 8, 14, 20)
                )
                save_prefix(worker)
                # Shared sources determine the schedule, not this window setting.
                worker.num_prefetch_layers = 1
                expected = (
                    [(0, 2), (2, 8), (8, 14), (14, 20), (23, 24)]
                    if bounded
                    else [(0, layer) for layer in range(3)]
                    + [(2, layer) for layer in range(3, 9)]
                    + [(8, layer) for layer in range(9, 15)]
                    + [(14, layer) for layer in range(15, 21)]
                    + [(20, 21), (21, 22), (22, 23), (23, 24)]
                )
                for _ in range(2):
                    submitted: list[tuple[int, int]] = []
                    enqueue = worker.kv_recv_thread.add_request

                    def record(task, worker=worker, submitted=submitted, enqueue=enqueue):
                        submitted.append((worker.current_layer, task.layer_id))
                        enqueue(task)

                    meta = metadata(worker, load=True, save=False)
                    worker.prepare_layerwise_step(meta)
                    with patch.object(worker.kv_recv_thread, "add_request", side_effect=record):
                        for layer in range(worker.num_layers):
                            forward_layer(worker, meta)
                    worker.get_finished(set(), meta)
                    self.assertEqual(submitted, expected)
                    self.assertFalse(store.leases)

    def test_source_load_waits_for_current_layer_not_whole_segment(self):
        for blocked_layer in (2, 8):
            with self.subTest(blocked_layer=blocked_layer):
                worker, store, caches = make_shared_worker(self, bounded=True)
                save_prefix(worker)
                store.block_address = caches[f"model.layers.{blocked_layer}.attn.long_kv_cache"][1].data_ptr()
                meta = metadata(worker, load=True, save=False)
                worker.prepare_layerwise_step(meta)
                passed_empty_layers = threading.Event()
                done = threading.Event()
                errors: list[Exception] = []

                def forward(worker, meta, passed_empty_layers, done, errors):
                    try:
                        for layer in range(3):
                            forward_layer(worker, meta)
                            if layer == 1:
                                passed_empty_layers.set()
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        done.set()

                thread = threading.Thread(target=forward, args=(worker, meta, passed_empty_layers, done, errors))
                thread.start()
                try:
                    self.assertTrue(store.blocked.wait(3))
                    self.assertTrue(passed_empty_layers.wait(3))
                    if blocked_layer == 2:
                        self.assertFalse(done.is_set())
                    else:
                        self.assertTrue(done.wait(3))
                    self.assertFalse(worker.layer_load_finished_events[blocked_layer].is_set())
                finally:
                    store.resume.set()
                    thread.join(3)
                self.assertTrue(done.is_set())
                self.assertFalse(errors)
                for layer in range(3, worker.num_layers):
                    forward_layer(worker, meta)
                worker.get_finished(set(), meta)

    def test_non_shared_model_keeps_configured_prefetch_window(self):
        worker, _, _ = make_shared_worker(self)
        worker.has_shared_kv = False
        worker.num_prefetch_layers = 2
        worker.layer_load_tasks = [[object()] for _ in range(worker.num_layers)]
        with patch.object(worker.kv_recv_thread, "add_request") as enqueue:
            worker._submit_ready_layer_loads()
            self.assertEqual([call.args[0].layer_id for call in enqueue.call_args_list], [0, 1, 2])
            worker.current_layer = 1
            worker._submit_ready_layer_loads()
            self.assertEqual([call.args[0].layer_id for call in enqueue.call_args_list], [0, 1, 2, 3])

    def test_cut_cache_only_load_is_idempotent_and_saves_after_final_write(self):
        worker, store, caches = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        store.published.clear()
        store.finish_calls.clear()
        meta = metadata(worker)
        conn = AscendStoreConnector.__new__(AscendStoreConnector)
        conn.use_layerwise = conn.can_put = True
        conn.connector_worker = worker
        conn._mamba_state = None
        conn.bind_connector_metadata(meta)
        for layer in range(8):
            name = f"model.layers.{layer}.v41_attn"
            conn.wait_for_layer_load(name)
            conn.save_kv_layer(name, [], None)
        cut_name = "model.layers.8.v41_attn"
        conn.wait_for_layer_load(cut_name)
        self.assertEqual(worker.current_layer, 8)
        conn.wait_for_layer_load(cut_name)
        cut = caches["model.layers.8.attn.long_kv_cache"]
        cut[1].fill_(77)
        conn.save_kv_layer(cut_name, [], None)
        worker.kv_send_thread.request_queue.join()
        self.assertTrue(any(cut[1].data_ptr() in addrs for direction, addrs in store.copies if direction == 0))
        self.assertEqual(worker.current_layer, 9)
        for layer in range(9, 12):
            name = f"model.layers.{layer}.v41_attn"
            conn.wait_for_layer_load(name)
            conn.save_kv_layer(name, [], None)
        self.assertTrue(store.published)

    def test_failed_load_reports_request_and_disables_saves(self):
        worker, store, _ = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        meta = metadata(worker, load=True, save=False)
        worker.prepare_layerwise_step(meta)
        store.fail_copy = True
        worker.start_load_kv(meta)
        for layer in range(12):
            forward_layer(worker, meta)
        self.assertEqual(worker.get_failed_requests(), {"r"})
        self.assertTrue(worker.kv_recv_thread.load_disabled.is_set())
        self.assertTrue(worker._save_disabled.is_set())
        self.assertFalse(meta.requests[0].can_save)

    def test_load_reuses_gva_hit_policy(self):
        for use_eagle in (False, True):
            with self.subTest(use_eagle=use_eagle):
                worker, store, _ = make_shared_worker(self, bounded=True)
                save_prefix(worker)
                worker.use_eagle = use_eagle
                meta = metadata(worker, load=True, save=False)
                # The last token is recomputed; the pool contains a full block.
                meta.requests[0].load_spec.kvpool_cached_tokens = 15
                meta.requests[0].load_spec.kvpool_store_skip_tokens = 16
                worker.prepare_layerwise_step(meta)
                if use_eagle:
                    self.assertFalse(worker.layer_load_tasks[2])
                else:
                    self.assertEqual(worker.layer_load_tasks[2][0].block_ranges[0].end_block, 1)
                copies_before_load = len(store.copies)
                for layer in range(worker.num_layers):
                    forward_layer(worker, meta)
                self.assertEqual(len(store.copies) > copies_before_load, not use_eagle)
                worker.get_finished(set(), meta)

    def test_shared_prefix_acquires_one_lease_per_step(self):
        worker, store, _ = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        counts: Counter[str] = Counter()
        add_lease, remove_lease = store.batch_add_lease, store.batch_remove_lease

        def acquire(keys, ttl):
            counts.update(keys)
            return add_lease(keys, ttl)

        def release(keys):
            counts.subtract(keys)
            return remove_lease(keys)

        store.batch_add_lease, store.batch_remove_lease = acquire, release
        meta = metadata(worker, load=True, save=False)
        other = deepcopy(meta.requests[0])
        other.req_id = "other"
        meta.requests.append(other)
        worker.prepare_layerwise_step(meta)
        self.assertEqual(set(counts.values()), {1})
        for layer in range(12):
            forward_layer(worker, meta)
        worker.get_finished(set(), meta)
        self.assertEqual(set(counts.values()), {0})

    def test_draft_layer_save_blocks_publication_until_final_copy(self):
        worker, store, caches = make_shared_worker(self, bounded=True, draft=True)
        meta = metadata(worker)
        worker.prepare_layerwise_step(meta)
        for layer in range(12):
            forward_layer(worker, meta)
        draft = caches["model.mtp.layers.0.attn.kv_cache"]
        draft[1].fill_(99)  # draft cache write before its save hook
        store.block_address = draft[1].data_ptr()
        store.block_direction = 0
        done = threading.Event()
        errors = []

        def save():
            try:
                forward_layer(worker, meta)
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=save)
        thread.start()
        try:
            self.assertTrue(store.blocked.wait(3))
            self.assertFalse(done.is_set())
            self.assertFalse(store.published)
        finally:
            store.resume.set()
            thread.join(3)
        self.assertTrue(done.is_set())
        self.assertFalse(errors)
        self.assertTrue(store.published)
        worker.get_finished(set(), meta)
        draft.zero_()
        meta = metadata(worker, load=True, save=False)
        worker.prepare_layerwise_step(meta)
        for layer in range(12):
            forward_layer(worker, meta)
        forward_layer(worker, meta)
        torch.testing.assert_close(draft[1], torch.full_like(draft[1], 99))

    def test_failed_draft_save_never_publishes_partial_prefix(self):
        worker, store, _ = make_shared_worker(self, bounded=True, draft=True)
        meta = metadata(worker)
        worker.prepare_layerwise_step(meta)
        for layer in range(12):
            forward_layer(worker, meta)
        store.fail_copy = True
        forward_layer(worker, meta)
        self.assertTrue(worker._save_disabled.is_set())
        self.assertFalse(store.published)
