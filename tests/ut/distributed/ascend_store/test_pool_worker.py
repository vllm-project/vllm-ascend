#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

import unittest
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
)


class TestKVPoolWorkerHelpers(unittest.TestCase):
    """Test the pure helper methods on KVPoolWorker without full init."""

    def _make_worker_class(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        return KVPoolWorker

    def test_check_all_layers_exists_all_present(self):
        cls = self._make_worker_class()
        # Manually call as unbound
        result = cls.check_all_layers_exists(None, [1, 1, 1, 1, 1, 1], 3)
        self.assertEqual(result, [1, 1])

    def test_check_all_layers_exists_partial(self):
        cls = self._make_worker_class()
        result = cls.check_all_layers_exists(None, [1, 1, 0, 1, 1, 1], 3)
        self.assertEqual(result, [0, 1])

    def test_check_all_layers_exists_none(self):
        cls = self._make_worker_class()
        result = cls.check_all_layers_exists(None, [0, 0, 0], 3)
        self.assertEqual(result, [0])

    def test_find_min_first_non_one_index_found(self):
        cls = self._make_worker_class()
        arr = [[1, 1, 0], [1, 0, 1]]
        result = cls.find_min_first_non_one_index(None, arr)
        self.assertEqual(result, 1)

    def test_find_min_first_non_one_index_all_one(self):
        cls = self._make_worker_class()
        arr = [[1, 1, 1], [1, 1, 1]]
        result = cls.find_min_first_non_one_index(None, arr)
        self.assertEqual(result, -1)

    def test_find_min_first_non_one_index_first_pos(self):
        cls = self._make_worker_class()
        arr = [[0, 1], [1, 0]]
        result = cls.find_min_first_non_one_index(None, arr)
        self.assertEqual(result, 0)

    def test_find_min_first_non_one_index_empty(self):
        cls = self._make_worker_class()
        result = cls.find_min_first_non_one_index(None, [])
        self.assertEqual(result, -1)


class TestKVPoolWorkerInit(unittest.TestCase):
    """Test KVPoolWorker initialization with mocked dependencies."""

    def _make_vllm_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])  # no index_topk
        config.model_config.get_num_layers.return_value = 32
        config.model_config.get_total_num_kv_heads.return_value = 8
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = block_size
        config.kv_events_config = None
        return config

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_basic(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                        mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        pcp_group.rank_in_group = 0
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0

        mock_backend = MagicMock()
        mock_importlib.import_module.return_value = mock_backend

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)

        self.assertEqual(worker.block_size, 16)
        self.assertEqual(worker.num_layers, 32)
        self.assertFalse(worker.use_layerwise)
        self.assertFalse(worker.use_mla)
        self.assertEqual(worker.tp_rank, 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_mla(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                      mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        config.model_config.use_mla = True
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        self.assertTrue(worker.use_mla)
        self.assertEqual(worker.num_kv_head, 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_kv_head_less_than_tp(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                                       mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 2
        mock_tp_size.return_value = 8
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        config.model_config.get_total_num_kv_heads.return_value = 4  # < tp_size=8
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        self.assertEqual(worker.put_step, 2)  # 8 / 4
        self.assertEqual(worker.head_or_tp_rank, 1)  # 2 // 2

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_get_kv_events_empty(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                                 mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        events = worker.get_kv_events()
        self.assertEqual(events, [])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_get_kv_events_with_send_thread(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                                            mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        config.kv_events_config = MagicMock()
        config.kv_events_config.enable_kv_cache_events = True
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.get_kv_events.return_value = [MagicMock()]
        events = worker.get_kv_events()
        self.assertEqual(len(events), 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_lookup_all_cached(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                               mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        worker.m_store.exists.return_value = [1, 1]
        result = worker.lookup(32, ["hash0", "hash1"], use_layerwise=False)
        self.assertEqual(result, 32)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_lookup_partial(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                            mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        worker.m_store.exists.return_value = [1, 0]
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)  # first non-exist at index 1 => starts[1]=16

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_lookup_exception(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                              mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        worker.m_store.exists.side_effect = Exception("conn error")
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_get_and_clear_finished_requests(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                                             mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)

        # Setup mock send thread using a real defaultdict
        from collections import defaultdict
        send_thread = MagicMock()
        stored = defaultdict(int)
        stored["r1"] = 0
        stored["r2"] = 1
        send_thread.stored_requests = stored
        worker.kv_send_thread = send_thread

        meta = AscendConnectorMetadata(set(), set())
        result = worker.get_and_clear_finished_requests({"r1"}, meta)
        self.assertIn("r1", result)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_consumer_partition_config(self, mock_tp_rank, mock_tp_size, mock_pcp_group,
                                       mock_dcp_ws, mock_dcp_rank, mock_importlib):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 1
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mock_pcp_group.return_value = pcp_group
        mock_dcp_ws.return_value = 1
        mock_dcp_rank.return_value = 0
        mock_importlib.import_module.return_value = MagicMock()

        config = self._make_vllm_config(
            kv_role="kv_consumer",
            extra_config={
                "backend": "mooncake",
                "consumer_is_to_put": True,
                "prefill_pp_layer_partition": "16,16",
                "prefill_pp_size": "2",
            },
        )
        config.model_config.hf_text_config.num_hidden_layers = 32
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        self.assertIsNotNone(worker.token_database.partitions)
        self.assertEqual(worker.token_database.partitions, [16, 16])


class TestKVPoolWorkerRegisterAndTransfer(unittest.TestCase):
    """Test register_kv_caches, start_load_kv, wait_for_save, get_finished, lookup_scheduler."""

    def _patch_all(self):
        """Return a dict of started patches."""
        patches = {
            "tp_rank": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank", return_value=0),
            "tp_size": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size", return_value=1),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size", return_value=1),
            "dcp_rank": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank", return_value=0),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()
        self._patches = patches
        return mocks

    def _stop_all(self):
        for p in self._patches.values():
            p.stop()

    def _make_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = block_size
        config.kv_events_config = None
        return config

    def _make_worker(self, kv_role="kv_producer", extra_config=None):
        self._patch_all()
        config = self._make_config(kv_role=kv_role, extra_config=extra_config)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        return worker

    def setUp(self):
        self._patches = {}

    def tearDown(self):
        self._stop_all()

    def test_register_kv_caches_non_mla(self):
        worker = self._make_worker()
        import torch
        fake_cache = MagicMock()
        fake_cache.shape = [100, 16, 8, 64]
        fake_cache.element_size.return_value = 2
        fake_cache.data_ptr.return_value = 10000
        kv_caches = {"layer.0": (fake_cache, fake_cache)}
        worker.register_kv_caches(kv_caches)
        self.assertEqual(len(worker.kv_caches_base_addr), 2)
        worker.m_store.register_buffer.assert_called_once()

    def test_start_load_kv_sync(self):
        worker = self._make_worker()
        worker.m_store.get = MagicMock()
        # Setup token database
        worker.token_database.set_kv_caches_base_addr([1000, 2000])
        worker.token_database.set_block_len([160])

        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=["h0"], load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        worker.m_store.get.assert_called_once()

    def test_start_load_kv_no_load(self):
        worker = self._make_worker()
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=["h0"], load_spec=None,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        # No get called since no load_spec

    def test_wait_for_save(self):
        worker = self._make_worker()
        worker.kv_send_thread = MagicMock()

        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=["h0"], can_save=True,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.wait_for_save(meta)
        worker.kv_send_thread.add_stored_request.assert_called_with("r1")
        worker.kv_send_thread.add_request.assert_called_once()

    def test_wait_for_save_skip_non_save(self):
        worker = self._make_worker()
        worker.kv_send_thread = MagicMock()

        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=["h0"], can_save=False,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.wait_for_save(meta)
        worker.kv_send_thread.add_stored_request.assert_not_called()

    def test_get_finished_producer(self):
        worker = self._make_worker(kv_role="kv_producer")
        from collections import defaultdict
        send_thread = MagicMock()
        stored = defaultdict(int)
        stored["r1"] = 0
        send_thread.stored_requests = stored
        worker.kv_send_thread = send_thread

        meta = AscendConnectorMetadata(set(), set())
        done_s, done_r = worker.get_finished({"r1"}, meta)
        self.assertIn("r1", done_s)
        self.assertEqual(done_r, set())

    def test_get_finished_consumer(self):
        worker = self._make_worker(kv_role="kv_consumer")
        meta = AscendConnectorMetadata(set(), set())
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())

    def test_lookup_scheduler_all_cached(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 1]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 32)

    def test_lookup_scheduler_partial(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 0]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)

    def test_lookup_scheduler_exception(self):
        worker = self._make_worker()
        worker.m_store.exists.side_effect = Exception("fail")
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    def test_lookup_layerwise(self):
        worker = self._make_worker()
        # 2 blocks * 2 layers = 4 keys, all exist
        worker.m_store.exists.return_value = [1, 1, 1, 1]
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=True)
        self.assertEqual(result, 32)

    def test_lookup_scheduler_layerwise(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 1, 1, 1]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=True)
        self.assertEqual(result, 32)

    def test_lookup_scheduler_multi_tp(self):
        self._stop_all()
        patches = {
            "tp_rank": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank", return_value=0),
            "tp_size": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size", return_value=2),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size", return_value=1),
            "dcp_rank": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank", return_value=0),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()
        self._patches = patches

        config = self._make_config()
        config.model_config.get_total_num_kv_heads.return_value = 2
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker(config, use_layerwize=False)
        # 2 blocks * 2 tp_ranks = 4 keys
        worker.m_store.exists.return_value = [1, 1, 1, 1]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 32)

    def test_get_and_clear_finished_requests_with_preempted(self):
        worker = self._make_worker()
        from collections import defaultdict
        send_thread = MagicMock()
        stored = defaultdict(int)
        stored["r1"] = 0
        send_thread.stored_requests = stored
        worker.kv_send_thread = send_thread

        meta = AscendConnectorMetadata(set(), {"r1"})
        result = worker.get_and_clear_finished_requests(set(), meta)
        send_thread.delete_finished_stored_request.assert_called_with("r1")

    def test_get_and_clear_finished_stored_req(self):
        worker = self._make_worker()
        from collections import defaultdict
        send_thread = MagicMock()
        stored = defaultdict(int)
        stored["r1"] = 0
        send_thread.stored_requests = stored
        worker.kv_send_thread = send_thread
        worker.finished_store_req.add("r1")

        meta = AscendConnectorMetadata(set(), set())
        result = worker.get_and_clear_finished_requests(set(), meta)
        self.assertIn("r1", result)

    def test_get_and_clear_finished_req_still_running(self):
        worker = self._make_worker()
        from collections import defaultdict
        send_thread = MagicMock()
        stored = defaultdict(int)
        stored["r1"] = 2  # still running
        send_thread.stored_requests = stored
        worker.kv_send_thread = send_thread

        meta = AscendConnectorMetadata(set(), set())
        result = worker.get_and_clear_finished_requests({"r1"}, meta)
        self.assertNotIn("r1", result)
        self.assertIn("r1", worker.finished_store_req)


if __name__ == "__main__":
    unittest.main()
