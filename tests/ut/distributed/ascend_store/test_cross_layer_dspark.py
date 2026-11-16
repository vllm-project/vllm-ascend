# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from itertools import product
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import torch

from tests.ut.distributed.ascend_store.test_cross_layer_gva import make_cross_worker, metadata, save_prefix


class TestDSparkCrossLayerGVA(unittest.TestCase):
    def test_v1_is_rejected_before_gva_initialization(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker.LayerwiseKVPoolWorker.__init__"
        ) as initialize_gva:
            for method in (None, "dspark"):
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "requires Model Runner V2"):
                    make_cross_worker(self, method=method, use_v2=False)
            initialize_gva.assert_not_called()

    def test_piecewise_is_rejected_before_gva_initialization(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker.LayerwiseKVPoolWorker.__init__"
        ) as initialize_gva:
            for mode in ("PIECEWISE", "FULL_AND_PIECEWISE"):
                for method in (None, "dspark"):
                    with self.subTest(mode=mode, method=method), self.assertRaisesRegex(ValueError, "--enforce-eager"):
                        make_cross_worker(self, mode=mode, method=method)
            initialize_gva.assert_not_called()

    def test_eager_keeps_target_pipeline(self):
        for method in (None, "dspark"):
            with self.subTest(method=method):
                worker, _, _ = make_cross_worker(self, method=method)
                self.assertTrue(worker._pipeline_enabled)

    def test_unnegotiated_modes_restore_synchronously_with_gva(self):
        for mode in ("FULL", "FULL_DECODE_ONLY", "UNKNOWN"):
            with self.subTest(mode=mode):
                worker, store, _ = make_cross_worker(self, mode=mode, draft=True)
                self.assertFalse(worker._pipeline_enabled)
                save_prefix(worker)
                meta = metadata(worker, load=True, save=False)
                worker.prepare_cross_layer_step(meta)
                worker.start_load_kv(meta)
                self.assertFalse(worker._pending_layers)
                worker.finish_cross_layer_loads()
                self.assertFalse(store.leases)

    def test_other_drafters_do_not_enable_pipeline(self):
        for method, architecture in (
            ("eagle", "DeepseekV41DSparkModel"),
            ("dspark", "OtherDraft"),
        ):
            with self.subTest(method=method, architecture=architecture):
                worker, store, _ = make_cross_worker(self, method=method, architecture=architecture)
                self.assertFalse(worker._pipeline_enabled)
                meta = metadata(worker)
                worker.prepare_cross_layer_step(meta)
                for layer in range(12):
                    worker.on_cross_layer_save_trigger(layer)
                worker.kv_send_thread.request_queue.join()
                self.assertFalse(store.copies)
                worker.wait_for_save(meta)
                self.assertTrue(store.published)

    def test_draft_restores_before_forward_while_target_loads_continue(self):
        worker, store, caches = make_cross_worker(self, bounded=True, draft=True, method="dspark")
        save_prefix(worker)
        for cache in caches.values():
            cache.zero_()
        store.block_address = caches["model.layers.8.attn.long_kv_cache"][1].data_ptr()
        meta = metadata(worker, load=True, save=False)
        worker.prepare_cross_layer_step(meta)
        worker.start_load_kv(meta)
        self.assertTrue(store.blocked.wait(timeout=2))
        self.assertGreater(int(caches["model.mtp.layers.0.attn.kv_cache"][1].sum()), 0)
        worker.wait_for_cross_layer_load(2)
        store.resume.set()
        worker.finish_cross_layer_loads()

    def test_dspark_target_saves_overlap_and_late_groups_use_final_contents(self):
        for bounded, (pcp_size, pcp_rank) in product((False, True), ((1, 0), (2, 0), (2, 1))):
            with self.subTest(bounded=bounded, pcp_size=pcp_size, pcp_rank=pcp_rank):
                worker, store, caches = make_cross_worker(
                    self, bounded=bounded, draft=True, method="dspark", pcp_size=pcp_size, pcp_rank=pcp_rank
                )
                meta = metadata(worker)
                worker.prepare_cross_layer_step(meta)
                for layer in range(12):
                    worker.on_cross_layer_save_trigger(layer)
                worker.kv_send_thread.request_queue.join()
                self.assertTrue(store.copies)
                self.assertFalse(store.published)
                copied = {address for direction, addresses in store.copies if direction == 0 for address in addresses}
                for name, cache in caches.items():
                    late = ".mtp." in name or (bounded and ".layers.8." in name)
                    self.assertEqual(cache[1].data_ptr() in copied, not late, name)
                    if late:
                        cache[1].fill_(37)
                expected = {name: cache[1].clone() for name, cache in caches.items()}
                worker.wait_for_save(meta)
                self.assertTrue(store.published)
                for cache in caches.values():
                    cache.zero_()
                restore = metadata(worker, load=True, save=False)
                worker.prepare_cross_layer_step(restore)
                worker.start_load_kv(restore)
                for layer in range(12):
                    worker.wait_for_cross_layer_load(layer)
                worker.finish_cross_layer_loads()
                for name, cache in caches.items():
                    torch.testing.assert_close(cache[1], expected[name])

    def test_target_sender_runs_while_later_layers_continue(self):
        worker, store, caches = make_cross_worker(self, bounded=True, draft=True, method="dspark")
        store.block_direction = 0
        store.block_address = caches["model.layers.2.attn.long_kv_cache"][1].data_ptr()
        meta = metadata(worker)
        worker.prepare_cross_layer_step(meta)
        worker.on_cross_layer_save_trigger(2)
        self.assertTrue(store.blocked.wait(timeout=2))
        # The producer thread proceeds while the original GVA sender is held.
        caches["model.layers.8.attn.long_kv_cache"][1].fill_(37)
        worker.on_cross_layer_save_trigger(8)
        self.assertFalse(worker.layer_save_finished_events[worker._global_to_local_layer[2]].is_set())
        self.assertFalse(store.published)
        store.resume.set()
        worker.wait_for_save(meta)
        self.assertTrue(store.published)

    def test_mixed_target_draft_group_without_mtp_prefix_stays_late(self):
        worker, store, _ = make_cross_worker(self, method="dspark")
        worker.kv_cache_config.dspark_draft_layer_names = ("model.layers.3.attn.swa_cache",)
        meta = metadata(worker)
        worker.prepare_cross_layer_step(meta)
        worker.on_cross_layer_save_trigger(3)
        worker.kv_send_thread.request_queue.join()
        self.assertFalse(store.copies)
        worker.wait_for_save(meta)
        self.assertTrue(store.published)

    def test_failed_early_target_save_never_publishes(self):
        worker, store, _ = make_cross_worker(self, bounded=True, draft=True, method="dspark")
        meta = metadata(worker)
        worker.prepare_cross_layer_step(meta)
        store.fail_copy = True
        worker.on_cross_layer_save_trigger(2)
        worker.kv_send_thread.request_queue.join()
        worker.wait_for_save(meta)
        self.assertFalse(store.published)
        self.assertFalse(store.finish_calls)
