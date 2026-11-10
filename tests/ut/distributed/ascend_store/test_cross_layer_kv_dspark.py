# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
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

    def test_eager_and_piecewise_keep_target_pipeline(self):
        for mode in ("NONE", "PIECEWISE"):
            for method in (None, "dspark"):
                with self.subTest(mode=mode, method=method):
                    worker, _, _ = make_cross_worker(self, mode=mode, method=method)
                    self.assertTrue(worker._pipeline_enabled)

    def test_unnegotiated_modes_restore_synchronously_with_gva(self):
        for mode in ("FULL", "FULL_DECODE_ONLY", "UNKNOWN"):
            with self.subTest(mode=mode):
                worker, store, _ = make_cross_worker(self, mode=mode, draft=True)
                self.assertFalse(worker._pipeline_enabled)
                save_prefix(worker)
                meta = metadata(worker, load=True, save=False)
                worker.prepare_cross_layer_kv_step(meta)
                worker.start_load_kv(meta)
                self.assertFalse(worker._pending_layers)
                worker.finish_cross_layer_kv_loads()
                self.assertFalse(store.leases)

    def test_other_drafters_do_not_enable_pipeline(self):
        for method, architecture in (
            ("eagle", "DeepseekV41DSparkModel"),
            ("dspark", "OtherDraft"),
        ):
            with self.subTest(method=method, architecture=architecture):
                worker, _, _ = make_cross_worker(self, method=method, architecture=architecture)
                self.assertFalse(worker._pipeline_enabled)

    def test_draft_restores_before_forward_while_target_loads_continue(self):
        worker, store, caches = make_cross_worker(self, bounded=True, draft=True, method="dspark")
        save_prefix(worker)
        for cache in caches.values():
            cache.zero_()
        store.block_address = caches["model.layers.8.attn.long_kv_cache"][1].data_ptr()
        meta = metadata(worker, load=True, save=False)
        worker.prepare_cross_layer_kv_step(meta)
        worker.start_load_kv(meta)
        self.assertTrue(store.blocked.wait(timeout=2))
        self.assertGreater(int(caches["model.mtp.layers.0.attn.kv_cache"][1].sum()), 0)
        worker.wait_for_cross_layer_kv_load(2)
        store.resume.set()
        worker.finish_cross_layer_kv_loads()

    def test_speculative_saves_use_final_cache_contents(self):
        worker, store, caches = make_cross_worker(self, bounded=True, draft=True, method="dspark")
        meta = metadata(worker)
        worker.prepare_cross_layer_kv_step(meta)
        for layer in range(12):
            worker.on_cross_layer_kv_save_trigger(layer)
        self.assertFalse(store.copies)
        for cache in caches.values():
            cache[1].fill_(37)
        worker.wait_for_save(meta)
        self.assertTrue(store.published)
        for key in store.published:
            address, size = store.objects[key]
            self.assertEqual(store.pool[address : address + size], bytes([37]) * size)
