# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-source forwards through Mooncake's range queues using CPU memory."""

import threading
import unittest
from unittest.mock import Mock, patch

import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from tests.ut.distributed.ascend_store.test_mooncake_hybrid import MemoryRangeStore
from tests.ut.distributed.ascend_store.test_shared_source_layerwise import make_shared_worker
from tests.ut.distributed.ascend_store.test_shared_source_layerwise import metadata as shared_metadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector


class SharedRangeStore(MemoryRangeStore):
    def __init__(self):
        super().__init__()
        self.resume = threading.Event()
        self.fail_put = False

    def ensure_initialized(self):
        pass

    def close(self):
        pass

    def exists(self, keys):
        return self.batch_is_exist(keys)

    def batch_copy_put(self, keys, buffers, sizes, offsets):
        if self.fail_put:
            return [-1] * len(keys)
        return super().batch_copy_put(keys, buffers, sizes, offsets)


def make_mooncake_worker(test, *, bounded, draft=False):
    store = SharedRangeStore()
    worker, _, caches = make_shared_worker(
        test, bounded=bounded, draft=draft, num_layers=40, sources=(2, 8, 14, 20), backend="mooncake", store=store
    )
    # A full, aligned page is reachable in each synthetic group. The real
    # coordinator/mask implementation is exercised by test_mooncake_hybrid.
    worker.cache_coordinator = object()
    masks = tuple([True] for _ in worker.grouped_block_size)
    worker.token_database.store_mask = Mock(return_value=masks)
    worker.token_database.load_mask = Mock(return_value=masks)
    conn = AscendStoreConnector.__new__(AscendStoreConnector)
    conn.use_layerwise = conn.can_put = True
    conn.connector_worker = worker
    conn._mamba_state = None
    return worker, store, caches, conn


def forward(conn, worker, *, cut=False):
    for layer in range(worker.num_layers):
        name = f"model.layers.{layer}.v41_attn" if layer < 40 else "model.mtp.layers.0.v41_attn"
        conn.wait_for_layer_load(name)
        if cut and layer == 20:
            # Cache-only entry and normal forward share the same load.
            conn.wait_for_layer_load(name)
        conn.save_kv_layer(name, [], None)


def metadata(worker, **kwargs):
    result = shared_metadata(worker, **kwargs)
    result.requests[0].is_last_chunk = True
    result.requests[0].num_prompt_tokens = 16
    return result


class TestMooncakeSharedSource(unittest.TestCase):
    def test_roundtrip_replay_on_off_and_draft(self):
        for bounded in (False, True):
            for draft in (False, True):
                with self.subTest(bounded=bounded, draft=draft):
                    worker, store, caches, conn = make_mooncake_worker(self, bounded=bounded, draft=draft)
                    self.assertTrue(worker.has_shared_kv)
                    self.assertEqual(worker.num_layers, 40 + int(draft))
                    expected = {name: cache.clone() for name, cache in caches.items()}
                    conn.bind_connector_metadata(metadata(worker))
                    forward(conn, worker, cut=True)
                    self.assertTrue(store.complete)
                    self.assertEqual(worker.current_layer, worker.num_layers)
                    self.assertFalse(worker._put_started_keys)
                    for cache in caches.values():
                        cache.zero_()
                    meta = metadata(worker, load=True, save=False)
                    submitted: list[tuple[int, int]] = []
                    enqueue = worker.kv_recv_thread.add_request

                    def record(task, worker=worker, submitted=submitted, enqueue=enqueue):
                        submitted.append((worker.current_layer, task.layer_id))
                        self.assertIsNone(task.attention_start_gate)
                        enqueue(task)

                    with patch.object(worker.kv_recv_thread, "add_request", side_effect=record):
                        conn.bind_connector_metadata(meta)
                        forward(conn, worker)
                    self.assertIn((0, 2), submitted)
                    self.assertIn((2, 8), submitted)
                    self.assertIn((8, 14), submitted)
                    self.assertIn((14, 20), submitted)
                    self.assertFalse(store.open_reads)
                    for name, cache in caches.items():
                        torch.testing.assert_close(cache[1], expected[name][1])
                        self.assertEqual(int(cache[0].sum()), 0)
                    # A later request sharing this prefix never reopens PUT on
                    # the already committed objects (the fake rejects that).
                    conn.bind_connector_metadata(metadata(worker))
                    forward(conn, worker)

    def test_shared_layer_save_hook_advances_execution(self):
        worker, _, _, conn = make_mooncake_worker(self, bounded=True)
        conn.bind_connector_metadata(metadata(worker))
        conn.wait_for_layer_load("model.layers.0.v41_attn")
        self.assertNotIn(0, worker._attention_saved_layers)
        self.assertEqual(worker.current_layer, 0)
        conn.save_kv_layer("model.layers.0.v41_attn", [], None)
        self.assertEqual(worker.current_layer, 1)
        for layer in range(1, worker.num_layers):
            conn.wait_for_layer_load(f"model.layers.{layer}.v41_attn")
            conn.save_kv_layer(f"model.layers.{layer}.v41_attn", [], None)

    def test_failed_put_revokes_uncommitted_objects(self):
        worker, store, _, conn = make_mooncake_worker(self, bounded=True)
        conn.bind_connector_metadata(metadata(worker))
        store.fail_put = True
        forward(conn, worker)
        worker.kv_send_thread.request_queue.join()
        self.assertFalse(store.complete)
        self.assertFalse(worker._put_started_keys)

    def test_failed_get_stops_before_attention_and_releases_sessions(self):
        worker, store, _, conn = make_mooncake_worker(self, bounded=True)
        conn.bind_connector_metadata(metadata(worker))
        forward(conn, worker)
        conn.bind_connector_metadata(metadata(worker, load=True, save=False))
        store.batch_copy_get = lambda keys, *args: [-1] * len(keys)
        with self.assertRaisesRegex(RuntimeError, "refusing incomplete"):
            forward(conn, worker)
        self.assertTrue(worker.get_block_ids_with_load_errors())
        self.assertFalse(store.open_reads)


if __name__ == "__main__":
    unittest.main()
