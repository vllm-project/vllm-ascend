#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LayerSaveTask,
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

    def test_find_all_continuous_hit_positions_found(self):
        cls = self._make_worker_class()
        arr = [[1, 1, 0], [1, 0, 1]]
        result = cls.find_all_continuous_hit_positions(arr, [16, 32, 48], 3, 48, 16)
        self.assertEqual(result, [16])

    def test_find_all_continuous_hit_positions_all_one(self):
        cls = self._make_worker_class()
        arr = [[1, 1, 1], [1, 1, 1]]
        result = cls.find_all_continuous_hit_positions(arr, [16, 32, 48], 3, 48, 16)
        self.assertEqual(result, [16, 32, 48])

    def test_find_all_continuous_hit_positions_first_pos(self):
        cls = self._make_worker_class()
        arr = [[0, 1], [1, 0]]
        result = cls.find_all_continuous_hit_positions(arr, [16, 32], 2, 48, 16)
        self.assertEqual(result, [])

    def test_find_all_continuous_hit_positions_empty(self):
        cls = self._make_worker_class()
        result = cls.find_all_continuous_hit_positions([], [], 0, 48, 16)
        self.assertEqual(result, [])

    def test_find_all_discontinuous_hit_positions_all_tp_hits(self):
        cls = self._make_worker_class()
        arr = [[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]]
        result = cls.find_all_discontinuous_hit_positions(arr, [16, 32, 48, 64, 80, 96], 6, 128, 16)
        self.assertEqual(result, [48, 96])

    def test_find_all_discontinuous_hit_positions_some_tp_hits(self):
        cls = self._make_worker_class()
        arr = [[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0]]
        result = cls.find_all_discontinuous_hit_positions(arr, [16, 32, 48, 64, 80, 96], 6, 128, 16)
        self.assertEqual(result, [48])

    def test_find_all_discontinuous_hit_positions_all_tp_hits_with_limits(self):
        cls = self._make_worker_class()
        arr = [[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]]
        result = cls.find_all_discontinuous_hit_positions(arr, [16, 32, 48, 64, 80, 96], 6, 64, 16)
        self.assertEqual(result, [48])

    def test_max_intersection_hit_position_single_group(self):
        cls = self._make_worker_class()
        hits = [[16, 32, 48]]
        self.assertEqual(48, cls._max_intersection_hit_position(hits))

    def test_max_intersection_hit_position_empty_group(self):
        cls = self._make_worker_class()
        hits: list[list[int]] = []
        self.assertEqual(0, cls._max_intersection_hit_position(hits))

    def test_max_intersection_hit_position_multi_group(self):
        cls = self._make_worker_class()
        hits = [[16, 32, 48], [32, 48], [16, 32], [32, 48, 64]]
        self.assertEqual(32, cls._max_intersection_hit_position(hits))

    def test_external_coordinator_lookup_disables_eagle_drop(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.num_kv_cache_groups = 1
        worker.cache_coordinator = MagicMock()
        worker.cache_coordinator.find_longest_cache_hit.return_value = ((), 128)
        worker.m_store = MagicMock()
        worker.m_store.exists.return_value = [1]

        key = MagicMock()
        key.chunk_hash = "ab" * 32
        key.to_string.return_value = "key"
        worker.token_database = MagicMock()
        worker.token_database.process_tokens.return_value = [(0, 128, key)]

        hit = worker._lookup_with_coordinator(
            128,
            [b"h0"],
            [0],
            use_layerwise=False,
            include_all_ranks=False,
        )

        self.assertEqual(hit, 128)
        worker.cache_coordinator.find_longest_cache_hit.assert_called_once()
        self.assertFalse(worker.cache_coordinator.find_longest_cache_hit.call_args.kwargs["apply_eagle"])


class TestKVPoolWorkerEarlyDispatch(unittest.TestCase):
    """Phase 2: on_kv_cache_written dispatches save at scatter, save_kv_layer falls back."""

    def _make_worker(self, num_layers=2, with_save_tasks=True):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = object.__new__(KVPoolWorker)
        worker.num_layers = num_layers
        worker.use_gva_layerwise = True
        worker.current_layer = 0
        worker._scatter_cursor = 0
        worker._early_dispatched = set()
        worker.sync_save_events = [MagicMock() for _ in range(num_layers)]
        worker.sync_attn_events = [MagicMock() for _ in range(num_layers)]
        worker.layer_attn_recorded_events = [threading.Event() for _ in range(num_layers)]
        worker.layer_save_finished_events = [threading.Event() for _ in range(num_layers)]
        worker.layer_save_tasks = [
            [MagicMock(block_ranges=[MagicMock(request=MagicMock(req_id="r1"))])] if with_save_tasks else []
            for _ in range(num_layers)
        ]
        worker.kv_send_thread = MagicMock()
        worker.prefetch_layer_map = {}
        worker.hf_config = MagicMock()
        worker.hf_config.num_hidden_layers = num_layers
        worker._layerwise_pd_transfer_waiter = None
        worker.layerwise_offload = False
        return worker

    def test_hook_dispatches_and_records_scatter_event(self):
        worker = self._make_worker()
        worker.on_kv_cache_written("model.layers.0.self_attn")
        worker.sync_save_events[0].record.assert_called_once()
        worker.kv_send_thread.add_request.assert_called_once()
        request = worker.kv_send_thread.add_request.call_args.args[0]
        self.assertIsInstance(request, LayerSaveTask)
        self.assertEqual(request.layer_id, 0)
        self.assertIs(request.transfer_tasks, worker.layer_save_tasks[0])
        self.assertIn(0, worker._early_dispatched)

    def test_hook_idempotent_on_repeat(self):
        worker = self._make_worker()
        worker.on_kv_cache_written("model.layers.0.self_attn")
        worker.on_kv_cache_written("model.layers.0.self_attn")
        worker.kv_send_thread.add_request.assert_called_once()
        worker.sync_save_events[0].record.assert_called_once()

    def test_hook_uses_cursor_when_layer_name_empty(self):
        worker = self._make_worker()
        worker.on_kv_cache_written("")
        # cursor resolves layer 0 and advances
        worker.sync_save_events[0].record.assert_called_once()
        self.assertEqual(worker._scatter_cursor, 1)

    def test_save_kv_layer_skips_already_dispatched(self):
        worker = self._make_worker()
        worker.on_kv_cache_written("model.layers.0.self_attn")
        worker.kv_send_thread.reset_mock()
        worker.save_kv_layer(MagicMock())
        # no second dispatch, but attn-done still recorded
        worker.kv_send_thread.add_request.assert_not_called()
        worker.sync_attn_events[0].record.assert_called_once()
        self.assertTrue(worker.layer_attn_recorded_events[0].is_set())
        self.assertEqual(worker.current_layer, 1)

    def test_save_kv_layer_fallback_dispatches_when_hook_missed(self):
        worker = self._make_worker()
        # If the attention path misses the hook, save_kv_layer must dispatch.
        worker.save_kv_layer(MagicMock())
        worker.sync_save_events[0].record.assert_called_once()
        worker.kv_send_thread.add_request.assert_called_once()
        worker.sync_attn_events[0].record.assert_called_once()
        self.assertIn(0, worker._early_dispatched)
        self.assertEqual(worker.current_layer, 1)

    def test_save_kv_layer_enqueues_control_task_when_hook_missed(self):
        worker = self._make_worker(with_save_tasks=False)
        worker.save_kv_layer(MagicMock())
        worker.kv_send_thread.add_request.assert_called_once()
        request = worker.kv_send_thread.add_request.call_args.args[0]
        self.assertIsInstance(request, LayerSaveTask)
        self.assertEqual(request.layer_id, 0)
        self.assertEqual(request.transfer_tasks, [])
        self.assertFalse(worker.layer_save_finished_events[0].is_set())
        self.assertTrue(worker.layer_attn_recorded_events[0].is_set())

    def test_hook_enqueues_control_task_for_layer_without_save_data(self):
        worker = self._make_worker(with_save_tasks=False)
        worker.on_kv_cache_written("model.layers.0.self_attn")

        worker.kv_send_thread.add_request.assert_called_once()
        request = worker.kv_send_thread.add_request.call_args.args[0]
        self.assertIsInstance(request, LayerSaveTask)
        self.assertEqual(request.layer_id, 0)
        self.assertEqual(request.transfer_tasks, [])
        self.assertIn(0, worker._early_dispatched)

        worker.save_kv_layer(MagicMock())
        worker.kv_send_thread.add_request.assert_called_once()
        self.assertTrue(worker.layer_attn_recorded_events[0].is_set())

    def test_key_path_keeps_raw_transfer_task_request(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = False

        worker.save_kv_layer(MagicMock())

        worker.kv_send_thread.add_request.assert_called_once_with(worker.layer_save_tasks[0])

    def test_key_path_completes_empty_layer_synchronously(self):
        worker = self._make_worker(with_save_tasks=False)
        worker.use_gva_layerwise = False

        worker.save_kv_layer(MagicMock())

        worker.kv_send_thread.add_request.assert_not_called()
        self.assertTrue(worker.layer_save_finished_events[0].is_set())


class TestKVPoolWorkerInit(unittest.TestCase):
    """Test KVPoolWorker initialization with mocked dependencies."""

    def _make_vllm_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])  # no index_topk
        config.model_config.get_num_layers.return_value = 32
        config.model_config.get_total_num_kv_heads.return_value = 8
        config.model_config.max_model_len = 1024
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = block_size
        config.kv_events_config = None
        return config

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_basic(self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib):
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

        worker = KVPoolWorker(config, use_layerwise=False)

        self.assertEqual(worker.block_size, 16)
        self.assertEqual(worker.num_layers, 32)
        self.assertFalse(worker.use_layerwise)
        self.assertFalse(worker.use_mla)
        self.assertEqual(worker.tp_rank, 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_mla(self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        self.assertTrue(worker.use_mla)
        self.assertEqual(worker.num_kv_head, 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_init_kv_head_less_than_tp(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        self.assertEqual(worker.put_step, 2)  # 8 / 4
        self.assertEqual(worker.head_or_tp_rank, 1)  # 2 // 2

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_get_kv_events_empty(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        events = worker.get_kv_events()
        self.assertEqual(events, [])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_get_kv_events_with_send_thread(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.get_kv_events.return_value = [MagicMock()]
        events = worker.get_kv_events()
        self.assertEqual(len(events), 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_lookup_all_cached(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        worker.m_store.exists.return_value = [1, 1]
        result = worker.lookup(32, ["hash0", "hash1"], use_layerwise=False)
        self.assertEqual(result, 32)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_lookup_partial(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        worker.m_store.exists.return_value = [1, 0]
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)  # first non-exist at index 1 => starts[1]=16

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_lookup_exception(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        worker.m_store.exists.side_effect = Exception("conn error")
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank")
    def test_consumer_partition_config(
        self, mock_tp_rank, mock_tp_size, mock_pcp_group, mock_dcp_ws, mock_dcp_rank, mock_importlib
    ):
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

        worker = KVPoolWorker(config, use_layerwise=False)
        self.assertIsNotNone(worker.token_database.partitions)
        self.assertEqual(worker.token_database.partitions, [16, 16])


class TestKVPoolWorkerRegisterAndTransfer(unittest.TestCase):
    """Test register_kv_caches, start_load_kv, wait_for_save, get_finished, lookup_scheduler."""

    def _patch_all(self):
        """Return a dict of started patches."""
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
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
        config.model_config.max_model_len = 1024
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

        worker = KVPoolWorker(config, use_layerwise=False)
        return worker

    def setUp(self):
        self._patches = {}

    def tearDown(self):
        self._stop_all()

    def test_register_kv_caches_non_mla(self):
        worker = self._make_worker()
        fake_cache = MagicMock()
        fake_cache.shape = [100, 16, 8, 64]
        fake_cache.element_size.return_value = 2
        fake_cache.data_ptr.return_value = 10000
        kv_caches = {"layer.0": (fake_cache, fake_cache)}
        # init_store + register_buffer now happen directly in register_kv_caches
        # (no separate init_backend handshake). Mark threads as already started
        # so we only exercise the buffer-registration path.
        worker._transfer_threads_started = True
        worker.register_kv_caches(kv_caches)
        self.assertEqual(len(worker.group_kv_caches_base_addr[0]), 2)
        self.assertEqual(worker.token_database.group_layer_offsets[0], [0, 2])
        worker.m_store.register_buffer.assert_called_once()

    def test_cache_group_metadata_bundles_optional_indexer_by_physical_layer(self):
        worker = self._make_worker()
        worker.num_blocks = 4
        worker.group_kv_caches_base_addr = {}
        worker.group_block_len = {}
        worker.group_block_stride = {}
        worker.group_layer_offsets = {}
        worker.group_num_layers = {}

        def make_cache(address, block_len):
            cache = MagicMock()
            cache.shape = [4, block_len]
            cache.__getitem__.return_value.numel.return_value = block_len
            cache.element_size.return_value = 1
            cache.stride.return_value = block_len
            cache.data_ptr.return_value = address
            return cache

        main0_k = make_cache(100, 4)
        main0_v = make_cache(200, 6)
        indexer0 = make_cache(250, 2)
        main1_k = make_cache(300, 4)
        main1_v = make_cache(400, 6)
        worker.kv_caches = {
            # Deliberately put the indexer first to verify layer-major ordering.
            "model.layers.0.self_attn.indexer.k_cache": (indexer0,),
            "model.layers.1.self_attn.attn": (main1_k, main1_v),
            "model.layers.0.self_attn.attn": (main0_k, main0_v),
        }

        worker._infer_cache_group_metadata(0, list(worker.kv_caches))

        self.assertEqual(worker.group_kv_caches_base_addr[0], [100, 200, 250, 300, 400])
        self.assertEqual(worker.group_block_len[0], [4, 6, 2, 4, 6])
        self.assertEqual(worker.group_layer_offsets[0], [0, 3, 5])
        self.assertEqual(worker.group_num_layers[0], 2)

    def test_cache_group_metadata_supports_sparse_c8_layer_layout(self):
        worker = self._make_worker()
        worker.num_blocks = 4
        worker.group_kv_caches_base_addr = {}
        worker.group_block_len = {}
        worker.group_block_stride = {}
        worker.group_layer_offsets = {}
        worker.group_num_layers = {}

        def make_cache(address, block_len):
            cache = MagicMock()
            cache.shape = [4, block_len]
            cache.__getitem__.return_value.numel.return_value = block_len
            cache.element_size.return_value = 1
            cache.stride.return_value = block_len
            cache.data_ptr.return_value = address
            return cache

        packed_main0 = make_cache(100, 12)
        indexer_k0 = make_cache(200, 4)
        indexer_scale0 = make_cache(300, 1)
        packed_main1 = make_cache(400, 12)
        worker.kv_caches = {
            "model.layers.0.self_attn.indexer.k_cache": (
                indexer_k0,
                indexer_scale0,
            ),
            "model.layers.1.self_attn.attn": (packed_main1,),
            "model.layers.0.self_attn.attn": (packed_main0,),
        }

        worker._infer_cache_group_metadata(0, list(worker.kv_caches))

        self.assertEqual(worker.group_kv_caches_base_addr[0], [100, 200, 300, 400])
        self.assertEqual(worker.group_block_len[0], [12, 4, 1, 12])
        self.assertEqual(worker.group_layer_offsets[0], [0, 3, 4])
        self.assertEqual(worker.group_num_layers[0], 2)

    def test_start_load_kv_sync(self):
        worker = self._make_worker()
        worker.m_store.get = MagicMock()
        # Setup token database
        worker.token_database.set_group_buffers({0: [1000, 2000]}, {0: [160]})

        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        worker.m_store.get.assert_called_once()

    def test_start_load_kv_sync_uses_tail_block_id(self):
        worker = self._make_worker()
        worker.m_store.get = MagicMock()
        worker.token_database.set_group_buffers({0: [1000]}, {0: [160]})

        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=64, can_load=True, token_len=64)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[99],
            block_hashes=["h0", "h1", "h2", "h3"],
            load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)

        worker.start_load_kv(meta)

        _, addrs, sizes = worker.m_store.get.call_args.args
        self.assertEqual(addrs, [[1000 + 99 * 160]])
        self.assertEqual(sizes, [[160]])

    def test_start_load_kv_no_load(self):
        worker = self._make_worker()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            load_spec=None,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        # No get called since no load_spec

    def test_wait_for_save(self):
        worker = self._make_worker()
        worker.kv_send_thread = MagicMock()

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            can_save=True,
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
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            can_save=False,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.wait_for_save(meta)
        worker.kv_send_thread.add_stored_request.assert_not_called()

    def test_get_finished_producer(self):
        worker = self._make_worker(kv_role="kv_producer")

        send_thread = MagicMock()
        send_thread.get_and_clear_finished_requests.return_value = {"r1"}
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
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
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

        worker = KVPoolWorker(config, use_layerwise=False)
        # 2 blocks * 2 tp_ranks = 4 keys
        worker.m_store.exists.return_value = [1, 1, 1, 1]
        result = worker.lookup_scheduler(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 32)


class TestKVPoolWorkerStaticHelpers(unittest.TestCase):
    """Test static and standalone helper methods."""

    def test_uses_hybrid_kv_cache_none_config(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        self.assertFalse(KVPoolWorker._uses_hybrid_kv_cache(MagicMock(), None))

    def test_uses_hybrid_kv_cache_disabled(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        vllm_config = MagicMock()
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_groups = [MagicMock()]
        self.assertFalse(KVPoolWorker._uses_hybrid_kv_cache(vllm_config, kv_cache_config))

    def test_uses_mamba_kv_cache_false_when_not_hybrid(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        self.assertFalse(KVPoolWorker._uses_mamba_kv_cache(False, None))

    def test_as_cache_tuple_tensor(self):
        import torch

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        t = torch.zeros(10)
        result = KVPoolWorker._as_cache_tuple(t)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], t)

    def test_as_cache_tuple_list(self):
        import torch

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        t1 = torch.zeros(10)
        t2 = torch.ones(10)
        result = KVPoolWorker._as_cache_tuple([t1, t2])
        self.assertEqual(len(result), 2)

    def test_get_group_family_out_of_range(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        self.assertEqual(KVPoolWorker._get_group_family(["a", "b"], 5), "default")

    def test_get_group_family_valid(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        self.assertEqual(KVPoolWorker._get_group_family(["a", "b"], 1), "b")


class TestKVPoolWorkerGetBlockIdsWithLoadErrors(unittest.TestCase):
    """Test get_block_ids_with_load_errors method."""

    def _make_worker(self):
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self._patches = patches
        return worker

    def tearDown(self):
        for p in self._patches.values():
            p.stop()

    def test_get_block_ids_with_load_errors_clears(self):
        worker = self._make_worker()
        worker._invalid_block_ids = {1, 2, 3}
        result = worker.get_block_ids_with_load_errors()
        self.assertEqual(result, {1, 2, 3})
        # Should be cleared after reading
        self.assertEqual(worker._invalid_block_ids, set())

    def test_get_block_ids_with_load_errors_empty(self):
        worker = self._make_worker()
        worker._invalid_block_ids = set()
        result = worker.get_block_ids_with_load_errors()
        self.assertEqual(result, set())


class TestKVPoolWorkerGetGroupTpSize(unittest.TestCase):
    """Test get_group_tp_size method."""

    def _make_worker(self):
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=4,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 8
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self._patches = patches
        return worker

    def tearDown(self):
        for p in self._patches.values():
            p.stop()

    def test_get_group_tp_size_align_state(self):
        worker = self._make_worker()
        worker.group_uses_align_state = [True]
        self.assertEqual(worker.get_group_tp_size(0), 4)

    def test_get_group_tp_size_normal(self):
        worker = self._make_worker()
        worker.group_uses_align_state = [False]
        self.assertEqual(worker.get_group_tp_size(0), 4)

    def test_get_group_tp_size_mla(self):
        worker = self._make_worker()
        worker.use_mla = True
        worker.group_uses_align_state = [False]
        # _get_group_num_kv_heads returns 1 for MLA
        self.assertEqual(worker.get_group_tp_size(0), 1)


class TestKVPoolWorkerBuildConnectorWorkerMeta(unittest.TestCase):
    """Test build_connector_worker_meta method."""

    def _make_worker(self):
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self._patches = patches
        return worker

    def tearDown(self):
        for p in self._patches.values():
            p.stop()

    def test_build_connector_worker_meta_non_mamba(self):
        worker = self._make_worker()
        worker.use_mamba = False
        self.assertIsNone(worker.build_connector_worker_meta())

    def test_build_connector_worker_meta_mamba_no_send_thread(self):
        worker = self._make_worker()
        worker.use_mamba = True
        worker.kv_send_thread = None
        self.assertIsNone(worker.build_connector_worker_meta())

    def test_build_connector_worker_meta_mamba_with_completed_events(self):
        worker = self._make_worker()
        worker.use_mamba = True

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread

        send_thread = MagicMock(spec=KVCacheStoreSendingThread)
        send_thread.get_completed_events.return_value = {1: 2}
        worker.kv_send_thread = send_thread

        result = worker.build_connector_worker_meta()
        self.assertIsNotNone(result)
        self.assertEqual(result.completed_events, {1: 2})

    def test_build_connector_worker_meta_mamba_no_completed_events(self):
        worker = self._make_worker()
        worker.use_mamba = True

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread

        send_thread = MagicMock(spec=KVCacheStoreSendingThread)
        send_thread.get_completed_events.return_value = {}
        worker.kv_send_thread = send_thread

        result = worker.build_connector_worker_meta()
        self.assertIsNone(result)


class TestKVPoolWorkerGetFinishedAsync(unittest.TestCase):
    """Test get_finished with async recv thread."""

    def _make_worker(self, kv_role="kv_consumer"):
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

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
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake", "load_async": True}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self._patches = patches
        return worker

    def tearDown(self):
        for p in self._patches.values():
            p.stop()

    def test_get_finished_async_with_recv_thread(self):
        worker = self._make_worker(kv_role="kv_consumer")
        worker.load_async = True

        recv_thread = MagicMock()
        recv_thread.get_and_clear_finished_requests.return_value = {"r1"}
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = None

        loading_req_ids = {"r1"}
        meta = AscendConnectorMetadata(set(), set(), loading_req_ids=loading_req_ids)
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, {"r1"})
        recv_thread.get_and_clear_finished_requests.assert_called_once_with(loading_req_ids)

    def test_get_finished_async_recv_discards_preempted(self):
        worker = self._make_worker(kv_role="kv_consumer")
        worker.load_async = True

        recv_thread = MagicMock()
        recv_thread.get_and_clear_finished_requests.return_value = set()
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = None

        meta = AscendConnectorMetadata(set(), {"r_preempted"}, loading_req_ids=set())
        worker.get_finished(set(), meta)
        recv_thread.discard_finished_requests.assert_called_once_with({"r_preempted"})

    def test_get_finished_layerwise_send_thread(self):
        worker = self._make_worker(kv_role="kv_producer")
        worker.use_layerwise = True

        send_thread = MagicMock()
        send_thread.get_and_clear_finished_requests.return_value = set()
        worker.kv_send_thread = send_thread
        worker.kv_recv_thread = None

        meta = AscendConnectorMetadata(set(), set())
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, set())
        send_thread.get_and_clear_finished_requests.assert_called_once_with()


class TestKVPoolWorkerInferGroupMethods(unittest.TestCase):
    """Test _infer_group_uses_align_state and _infer_group_block_sizes."""

    def test_infer_group_uses_align_state_no_config(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        worker = KVPoolWorker(config, use_layerwise=False)
        self.assertEqual(worker.group_uses_align_state, [False])

        for p in patches.values():
            p.stop()

    def test_get_group_block_size_out_of_range(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        worker = KVPoolWorker(config, use_layerwise=False)
        # group_id out of range returns first element
        self.assertEqual(worker._get_group_block_size(5), 16)

        for p in patches.values():
            p.stop()


class TestKVPoolWorkerStartLoadKVAsync(unittest.TestCase):
    """Test start_load_kv with load_async=True."""

    def _make_worker(self):
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = "kv_consumer"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake", "load_async": True}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        worker.load_async = True
        self._patches = patches
        return worker

    def tearDown(self):
        for p in self._patches.values():
            p.stop()

    def test_start_load_kv_async_delegates_to_recv_thread(self):
        worker = self._make_worker()
        recv_thread = MagicMock()
        worker.kv_recv_thread = recv_thread

        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        recv_thread.add_request.assert_called_once_with(req)

    def test_start_load_kv_empty_requests(self):
        worker = self._make_worker()
        meta = AscendConnectorMetadata(set(), set())
        worker.start_load_kv(meta)
        # No action taken, no error


class TestKVPoolWorkerProcessLayerData(unittest.TestCase):
    """Test process_layer_data and related layerwise methods."""

    def _make_worker(self):
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            "pcp_group": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            "dcp_ws": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            "dcp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            "importlib": patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        }
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks["pcp_group"].return_value = pcp_group
        mocks["importlib"].import_module.return_value = MagicMock()

        config = MagicMock()
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_num_layers.return_value = 2
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=False)
        self._patches = patches
        return worker

    @staticmethod
    def _configure_layerwise_transfer_preparer(worker):
        preparer = worker._layerwise_transfer_preparer
        preparer.enabled = worker.use_gva_layerwise
        preparer.can_allocate = (
            worker.kv_role != "kv_consumer" or worker.consumer_is_to_put
        ) and worker.tp_rank % worker.put_step == 0
        preparer.num_groups = worker.num_kv_cache_groups
        preparer.configure_layout(
            worker.group_block_len,
        )
        return preparer

    @classmethod
    def _allocate_save_batches(cls, worker, plans):
        preparer = cls._configure_layerwise_transfer_preparer(worker)
        return preparer.resolve_save_groups(plans)

    @classmethod
    def _prepare_load_batches(cls, worker, plans):
        preparer = cls._configure_layerwise_transfer_preparer(worker)
        prepared = preparer.resolve_load_groups(plans)
        return {group_id: resolved for (group_id, uses_hbm_tail), resolved in prepared.items() if not uses_hbm_tail}

    def tearDown(self):
        for p in self._patches.values():
            p.stop()

    def test_process_layer_data_empty_requests(self):
        worker = self._make_worker()
        worker.process_layer_data([])
        # layer tasks should remain empty
        for layer_tasks in worker.layer_save_tasks:
            self.assertEqual(len(layer_tasks), 0)
        for layer_tasks in worker.layer_load_tasks:
            self.assertEqual(len(layer_tasks), 0)

    def test_group_batch_plan_skips_no_save(self):
        worker = self._make_worker()
        req = ReqMeta(req_id="r1", token_len_chunk=32, block_ids=[0, 1], block_hashes=["h0", "h1"], can_save=False)
        plans = worker._build_group_batch_plans([req])
        self.assertEqual(plans[0].save_ranges, [])

    def test_group_batch_plan_skips_zero_save_range(self):
        worker = self._make_worker()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            can_save=True,
            save_start_token=16,
            save_end_token=16,
        )
        plans = worker._build_group_batch_plans([req])
        self.assertEqual(plans[0].save_ranges, [])

    def test_group_batch_plan_skips_no_load(self):
        worker = self._make_worker()
        req = ReqMeta(req_id="r1", token_len_chunk=32, block_ids=[0, 1], block_hashes=["h0", "h1"], load_spec=None)
        plans = worker._build_group_batch_plans([req])
        self.assertEqual(plans[0].full_load_ranges, [])

    def test_group_batch_plan_skips_cannot_load(self):
        worker = self._make_worker()
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=0, can_load=False, token_len=0)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=load_spec,
        )
        plans = worker._build_group_batch_plans([req])
        self.assertEqual(plans[0].full_load_ranges, [])

    def test_group_batch_plans_group_requests_before_transfer(self):
        worker = self._make_worker()
        worker.num_kv_cache_groups = 2
        worker.grouped_block_size = [16, 16]
        worker.kv_cache_group_families = ["default", "default"]
        requests = [
            ReqMeta(
                req_id=f"r{request_index}",
                token_len_chunk=16,
                block_hashes=[f"h{request_index}"],
                can_save=True,
            )
            for request_index in range(2)
        ]

        plans = worker._build_group_batch_plans(requests)

        self.assertEqual([plan.group_id for plan in plans], [0, 1])
        self.assertEqual(
            [[block_range.request.req_id for block_range in plan.save_ranges] for plan in plans],
            [["r0", "r1"], ["r0", "r1"]],
        )

    def test_process_layer_data_reuses_group_batch_plan_across_layers(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.kv_recv_thread = MagicMock()
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            can_save=True,
            load_spec=load_spec,
        )

        preparer = worker._layerwise_transfer_preparer
        with (
            patch.object(preparer, "resolve_save_groups") as allocate,
            patch.object(preparer, "resolve_load_groups") as prepare_load,
        ):
            worker.process_layer_data([req])

        allocate.assert_not_called()
        prepare_load.assert_not_called()
        self.assertIs(
            worker.layer_save_tasks[0][0].block_ranges,
            worker.layer_save_tasks[1][0].block_ranges,
        )
        self.assertIs(
            worker.layer_load_tasks[0][0].block_ranges,
            worker.layer_load_tasks[1][0].block_ranges,
        )

    def test_process_layer_data_defers_preparation_until_transfer_thread(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        self._configure_layerwise_transfer_preparer(worker)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            can_save=True,
        )

        preparer = worker._layerwise_transfer_preparer
        resolved = (MagicMock(), MagicMock())
        with patch.object(preparer, "resolve_save_groups", return_value={0: resolved}) as alloc:
            worker.process_layer_data([req])
            preparation = worker.layer_save_tasks[0][0].preparation
            self.assertIsNotNone(preparation)
            alloc.assert_not_called()

            preparation.ensure_ready()
            preparation.ensure_ready()

        alloc.assert_called_once()

    def test_layer_reuse_loads_full_prefix_only_for_shared_layers(self):
        worker = self._make_worker()
        worker.num_layers = 3
        worker.layerwise_offload = True
        worker.independent_layers = [0, 2]
        load_spec = LoadSpec(
            vllm_cached_tokens=16,
            kvpool_cached_tokens=32,
            can_load=True,
            token_len=32,
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=load_spec,
        )

        worker.process_layer_data([req])

        self.assertEqual(worker.layer_load_tasks[0][0].block_ranges[0].start_block, 1)
        self.assertEqual(worker.layer_load_tasks[1][0].block_ranges[0].start_block, 0)
        self.assertEqual(worker.layer_load_tasks[2][0].block_ranges[0].start_block, 1)
        self.assertIs(
            worker.layer_load_tasks[0][0].block_ranges,
            worker.layer_load_tasks[2][0].block_ranges,
        )

    def test_layer_reuse_slices_each_request_by_its_hbm_prefix(self):
        worker = self._make_worker()
        worker.layerwise_offload = True
        worker.independent_layers = [0]
        requests = [
            ReqMeta(
                req_id="tail",
                token_len_chunk=32,
                block_ids=[0, 1],
                block_hashes=["h0", "h1"],
                load_spec=LoadSpec(
                    vllm_cached_tokens=16,
                    kvpool_cached_tokens=32,
                    can_load=True,
                ),
            ),
            ReqMeta(
                req_id="fully-local",
                token_len_chunk=32,
                block_ids=[2, 3],
                block_hashes=["h2", "h3"],
                load_spec=LoadSpec(
                    vllm_cached_tokens=32,
                    kvpool_cached_tokens=32,
                    can_load=True,
                ),
            ),
        ]

        worker.process_layer_data(requests)

        independent_ranges = worker.layer_load_tasks[0][0].block_ranges
        shared_ranges = worker.layer_load_tasks[1][0].block_ranges
        self.assertEqual(
            [
                (block_range.request.req_id, block_range.start_block, block_range.end_block)
                for block_range in independent_ranges
            ],
            [("tail", 1, 2)],
        )
        self.assertEqual(
            [
                (block_range.request.req_id, block_range.start_block, block_range.end_block)
                for block_range in shared_ranges
            ],
            [("tail", 0, 2), ("fully-local", 0, 2)],
        )

    def test_reused_layer_without_load_still_submits_save_gate(self):
        worker = self._make_worker()
        worker.num_layers = 3
        worker.current_layer = 2
        worker.next_layer_to_submit = 2
        worker.num_prefetch_layers = 1
        worker.prefetch_layer_map = {2: 0}
        worker.layer_load_tasks = [[], [], []]
        worker.kv_recv_thread = MagicMock()

        worker._submit_ready_layer_loads()

        task = worker.kv_recv_thread.add_request.call_args.args[0]
        self.assertEqual(task.layer_id, 2)
        self.assertEqual(task.wait_for_save_layer, 0)
        self.assertEqual(task.transfer_tasks, [])

    def test_final_save_keeps_reuse_gate_for_receive_thread(self):
        worker = self._make_worker()
        worker.num_layers = 3
        worker.current_layer = 2
        worker.prefetch_layer_map = {2: 0}
        worker.layer_save_tasks = [[], [], []]
        worker.sync_save_events = [MagicMock() for _ in range(3)]
        worker.layer_save_finished_events = [threading.Event() for _ in range(3)]
        worker.layer_save_finished_events[0].set()
        worker.layer_save_finished_events[1].set()
        worker.kv_send_thread = MagicMock()

        worker.save_kv_layer(MagicMock())

        # Layer 0 gates reuse by layer 2, so only the receive thread may
        # consume it. Unrelated and final-layer events are cleaned here.
        self.assertTrue(worker.layer_save_finished_events[0].is_set())
        self.assertFalse(worker.layer_save_finished_events[1].is_set())
        self.assertFalse(worker.layer_save_finished_events[2].is_set())

    def test_empty_layer_save_waits_for_pd_before_completion(self):
        worker = self._make_worker()
        worker.num_layers = 2
        worker.current_layer = 0
        worker.layerwise_offload = True
        worker.prefetch_layer_map = {1: 0}
        worker.layer_save_tasks = [[], []]
        worker.sync_save_events = [MagicMock(), MagicMock()]
        worker.layer_save_finished_events = [threading.Event(), threading.Event()]
        layer_finished = worker.layer_save_finished_events[0]

        def wait_for_pd(layer_id):
            self.assertEqual(layer_id, 0)
            self.assertFalse(layer_finished.is_set())

        worker._layerwise_pd_transfer_waiter = MagicMock(side_effect=wait_for_pd)
        worker.kv_send_thread = MagicMock()

        worker.save_kv_layer(MagicMock())

        worker._layerwise_pd_transfer_waiter.assert_called_once_with(0)
        self.assertTrue(layer_finished.is_set())

    def test_alloc_gvas_for_save_stores_only_planned_range(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.group_num_layers = {0: 2}
        worker.group_block_len = {0: [16, 16]}
        worker.page_size_bytes = 32
        worker.m_store.batch_alloc.return_value = [900, 901]
        block_ids = np.arange(10, dtype=np.int64)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=160,
            save_start_token=128,
            save_end_token=160,
            block_ids=list(range(10)),
            block_ids_by_group_np=[block_ids],
            block_hashes=[f"h{i}" for i in range(10)],
            can_save=True,
        )
        plans = worker._build_group_batch_plans([req])

        batches = self._allocate_save_batches(worker, plans)

        data, _ = batches[0]
        np.testing.assert_array_equal(data.block_ids_arr, [8, 9])
        np.testing.assert_array_equal(data.base_gvas_arr, [900, 901])

    def test_alloc_gvas_for_save_batches_requests(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.group_num_layers = {0: 2}
        worker.group_block_len = {0: [16, 16]}
        worker.page_size_bytes = 32
        requests = [
            ReqMeta(
                req_id=f"r{index}",
                token_len_chunk=16,
                save_end_token=16,
                block_ids=[index],
                block_ids_by_group_np=[np.asarray([index], dtype=np.int64)],
                block_hashes=[f"h{index}"],
                can_save=True,
            )
            for index in range(2)
        ]
        plans = worker._build_group_batch_plans(requests)
        worker.m_store.batch_alloc.return_value = [900, 901]

        batches = self._allocate_save_batches(worker, plans)

        worker.m_store.batch_alloc.assert_called_once()
        alloc_keys, alloc_sizes = worker.m_store.batch_alloc.call_args.args
        self.assertEqual(len(alloc_keys), 2)
        self.assertEqual(alloc_sizes, [32, 32])
        data, completion = batches[0]
        np.testing.assert_array_equal(data.block_ids_arr, [0, 1])
        np.testing.assert_array_equal(data.base_gvas_arr, [900, 901])
        self.assertEqual(completion.req_ids, ["r0", "r1"])

    def test_alloc_gvas_for_save_deduplicates_shared_keys_per_group(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.group_num_layers = {0: 1}
        worker.group_block_len = {0: [16]}
        requests = [
            ReqMeta(
                req_id="r0",
                token_len_chunk=32,
                block_ids=[10, 11],
                block_hashes=["shared", "h0"],
                can_save=True,
            ),
            ReqMeta(
                req_id="r1",
                token_len_chunk=32,
                block_ids=[20, 21],
                block_hashes=["shared", "h1"],
                can_save=True,
            ),
        ]
        plans = worker._build_group_batch_plans(requests)
        worker.m_store.batch_alloc.return_value = [900, 901, 902]

        batches = self._allocate_save_batches(worker, plans)

        alloc_keys, alloc_sizes = worker.m_store.batch_alloc.call_args.args
        self.assertEqual(len(alloc_keys), 3)
        self.assertEqual(alloc_sizes, [16, 16, 16])
        data, completion = batches[0]
        np.testing.assert_array_equal(data.block_ids_arr, [10, 11, 21])
        np.testing.assert_array_equal(data.base_gvas_arr, [900, 901, 902])
        self.assertEqual(completion.req_ids, ["r0", "r1"])

    def test_alloc_gvas_for_save_rejects_invalid_gva(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.group_num_layers = {0: 1}
        worker.group_block_len = {0: [16]}
        worker.page_size_bytes = 16
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_ids_by_group_np=[np.asarray([0], dtype=np.int64)],
            block_hashes=["h0"],
            can_save=True,
        )
        plans = worker._build_group_batch_plans([request])
        worker.m_store.batch_alloc.return_value = [0]

        with self.assertRaisesRegex(RuntimeError, "invalid GVAs"):
            self._allocate_save_batches(worker, plans)

    def test_chunked_prefill_loads_previous_chunk_gvas_by_key(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.layerwise_offload = True
        worker.group_num_layers = {0: 2}
        worker.group_block_len = {0: [16, 16]}
        worker.page_size_bytes = 32

        first_chunk = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids=[0, 1],
            block_ids_by_group_np=[np.arange(2, dtype=np.int64)],
            block_hashes=["h0", "h1"],
            can_save=True,
        )
        first_plans = worker._build_group_batch_plans([first_chunk])
        worker.m_store.batch_alloc.return_value = [800, 801]
        self._allocate_save_batches(worker, first_plans)
        first_keys = worker.m_store.batch_alloc.call_args.args[0]

        second_chunk = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            save_start_token=32,
            save_end_token=64,
            block_ids=[0, 1, 2, 3],
            block_ids_by_group_np=[np.arange(4, dtype=np.int64)],
            block_hashes=["h0", "h1", "h2", "h3"],
            can_save=True,
            load_spec=LoadSpec(
                vllm_cached_tokens=32,
                kvpool_cached_tokens=32,
                can_load=True,
                token_len=64,
            ),
        )
        second_plans = worker._build_group_batch_plans([second_chunk])
        self.assertEqual(second_plans[0].full_load_ranges[0].start_block, 0)
        self.assertEqual(second_plans[0].full_load_ranges[0].end_block, 2)
        self.assertEqual(second_plans[0].save_ranges[0].start_block, 2)
        self.assertEqual(second_plans[0].save_ranges[0].end_block, 4)

        key_infos = []
        for gva in (800, 801):
            key_info = MagicMock()
            key_info.size.return_value = 1
            key_info.gva_list.return_value = [gva]
            key_infos.append(key_info)
        worker.m_store.batch_get_key_info.return_value = key_infos
        worker.m_store.batch_add_lease.return_value = [0, 0]

        batches = self._prepare_load_batches(worker, second_plans)

        worker.m_store.batch_get_key_info.assert_called_once_with(first_keys, flag=1)
        np.testing.assert_array_equal(batches[0][0].base_gvas_arr, [800, 801])

    def test_prepare_load_gvas_stores_only_planned_range(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        key_infos = []
        for gva in (800, 801):
            key_info = MagicMock()
            key_info.size.return_value = 1
            key_info.gva_list.return_value = [gva]
            key_infos.append(key_info)
        worker.m_store.batch_get_key_info.return_value = key_infos
        worker.m_store.batch_add_lease.return_value = [0, 0]
        block_ids = np.arange(10, dtype=np.int64)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=160,
            block_ids=list(range(10)),
            block_ids_by_group_np=[block_ids],
            block_hashes=[f"h{i}" for i in range(10)],
            load_spec=LoadSpec(vllm_cached_tokens=128, kvpool_cached_tokens=160, can_load=True, token_len=160),
        )
        plans = worker._build_group_batch_plans([req])

        batches = self._prepare_load_batches(worker, plans)

        data, _ = batches[0]
        np.testing.assert_array_equal(data.block_ids_arr, [8, 9])
        np.testing.assert_array_equal(data.base_gvas_arr, [800, 801])

    def test_prepare_load_gvas_batches_requests_and_tracks_unique_keys(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        requests = [
            ReqMeta(
                req_id=f"r{index}",
                token_len_chunk=32,
                block_ids=[0, 1],
                block_ids_by_group_np=[np.arange(2, dtype=np.int64)],
                block_hashes=["shared", f"h{index}"],
                load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32),
            )
            for index in range(2)
        ]
        plans = worker._build_group_batch_plans(requests)
        unique_key_count = 3
        key_infos = []
        for gva in range(700, 700 + unique_key_count):
            key_info = MagicMock()
            key_info.size.return_value = 1
            key_info.gva_list.return_value = [gva]
            key_infos.append(key_info)
        worker.m_store.batch_get_key_info.return_value = key_infos
        worker.m_store.batch_add_lease.return_value = [0] * unique_key_count
        batches = self._prepare_load_batches(worker, plans)

        unique_keys = worker.m_store.batch_get_key_info.call_args.args[0]
        worker.m_store.batch_get_key_info.assert_called_once_with(unique_keys, flag=1)
        worker.m_store.batch_add_lease.assert_called_once_with(unique_keys, 5 * 60 * 1000)
        self.assertEqual(worker._layerwise_transfer_preparer.load_lease_refcounts[unique_keys[0]], 2)
        self.assertEqual(len(unique_keys), 3)
        data, completion = batches[0]
        np.testing.assert_array_equal(data.block_ids_arr, [0, 1, 0, 1])
        np.testing.assert_array_equal(data.base_gvas_arr, [700, 701, 700, 702])
        self.assertEqual(completion.req_ids, ["r0", "r1"])

    def test_prepare_load_gvas_reuses_full_batch_and_builds_group_tail(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.layerwise_offload = True
        requests = [
            ReqMeta(
                req_id="tail",
                token_len_chunk=32,
                block_ids=[0, 1],
                block_hashes=["h0", "h1"],
                load_spec=LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=True),
            ),
            ReqMeta(
                req_id="fully-local",
                token_len_chunk=32,
                block_ids=[2, 3],
                block_hashes=["h2", "h3"],
                load_spec=LoadSpec(vllm_cached_tokens=32, kvpool_cached_tokens=32, can_load=True),
            ),
        ]
        plans = worker._build_group_batch_plans(requests)
        key_infos = []
        for gva in (700, 701, 702, 703):
            key_info = MagicMock()
            key_info.size.return_value = 1
            key_info.gva_list.return_value = [gva]
            key_infos.append(key_info)
        worker.m_store.batch_get_key_info.return_value = key_infos
        worker.m_store.batch_add_lease.return_value = [0, 0, 0, 0]
        preparer = self._configure_layerwise_transfer_preparer(worker)

        prepared = preparer.resolve_load_groups(plans)
        full_data, _ = prepared[(0, False)]
        tail_data, tail_completion = prepared[(0, True)]

        np.testing.assert_array_equal(full_data.block_ids_arr, [0, 1, 2, 3])
        np.testing.assert_array_equal(full_data.base_gvas_arr, [700, 701, 702, 703])
        np.testing.assert_array_equal(tail_data.block_ids_arr, [1])
        np.testing.assert_array_equal(tail_data.base_gvas_arr, [701])
        self.assertEqual(tail_completion.req_ids, ["tail"])

    def test_flat_gva_batches_merge_all_groups(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        worker.grouped_block_size = [16, 16]
        worker.num_kv_cache_groups = 2
        worker.kv_cache_group_families = ["default", "default"]
        worker.group_num_layers = {0: 1, 1: 1}
        worker.group_block_len = {0: [16], 1: [16]}
        block_ids_by_group = [
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([10, 11], dtype=np.int64),
        ]
        save_request = ReqMeta(
            req_id="save",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1], [10, 11]],
            block_ids_by_group_np=block_ids_by_group,
            block_hashes=["h0", "h1"],
            can_save=True,
        )
        save_plans = worker._build_group_batch_plans([save_request])
        worker.m_store.batch_alloc.return_value = [100, 101, 200, 201]

        save_batches = self._allocate_save_batches(worker, save_plans)

        worker.m_store.batch_alloc.assert_called_once()
        self.assertEqual(len(worker.m_store.batch_alloc.call_args.args[0]), 4)
        np.testing.assert_array_equal(save_batches[0][0].base_gvas_arr, [100, 101])
        np.testing.assert_array_equal(save_batches[1][0].base_gvas_arr, [200, 201])

        load_request = ReqMeta(
            req_id="load",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1], [10, 11]],
            block_ids_by_group_np=block_ids_by_group,
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True),
        )
        load_plans = worker._build_group_batch_plans([load_request])
        key_infos = []
        for gva in (100, 101, 200, 201):
            key_info = MagicMock()
            key_info.size.return_value = 1
            key_info.gva_list.return_value = [gva]
            key_infos.append(key_info)
        worker.m_store.batch_get_key_info.return_value = key_infos
        worker.m_store.batch_add_lease.return_value = [0, 0, 0, 0]

        load_batches = self._prepare_load_batches(worker, load_plans)

        worker.m_store.batch_get_key_info.assert_called_once()
        self.assertEqual(len(worker.m_store.batch_get_key_info.call_args.args[0]), 4)
        np.testing.assert_array_equal(load_batches[0][0].base_gvas_arr, [100, 101])
        np.testing.assert_array_equal(load_batches[1][0].base_gvas_arr, [200, 201])

    def test_finished_request_releases_only_unshared_leases(self):
        worker = self._make_worker()
        worker.use_gva_layerwise = True
        self._configure_layerwise_transfer_preparer(worker)
        worker.m_store.batch_remove_lease.return_value = 0
        worker._layerwise_transfer_preparer.register_load_leases({"r1": {"shared", "only-r1"}, "r2": {"shared"}})

        worker.get_finished({"r1"}, AscendConnectorMetadata(set(), set()))

        worker.m_store.batch_remove_lease.assert_called_once_with(["only-r1"])
        worker.m_store.batch_remove_lease.reset_mock()

        worker.get_finished({"r2"}, AscendConnectorMetadata(set(), set()))

        worker.m_store.batch_remove_lease.assert_called_once_with(["shared"])


class TestKVPoolWorkerTpMismatch(unittest.TestCase):
    """Tests for TP-asymmetric prefill/decode strided KV transfer.

    Scenario: decode node (tp2) stores KV, prefill node (tp4) loads/hits.
    Qwen3-8B GQA: num_kv_heads=8 -> decode tp2 holds 4 heads/rank, prefill tp4
    holds 2 heads/rank; effective_tp=4, decode num_sub_keys=2.
    """

    def _make_vllm_config(self, kv_role="kv_consumer", extra_config=None, num_kv_heads=8, use_sparse=False):
        config = MagicMock()
        config.model_config.model = "qwen/qwen3-8b"
        config.model_config.use_mla = False
        if use_sparse:
            config.model_config.hf_text_config = MagicMock()
            config.model_config.hf_text_config.index_topk = 32
        else:
            config.model_config.hf_text_config = MagicMock(spec=[])  # no index_topk
        config.model_config.get_num_layers.return_value = 36
        config.model_config.get_total_num_kv_heads.return_value = num_kv_heads
        config.model_config.max_model_len = 4096
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.rank = 0
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = 16
        config.kv_events_config = None
        return config

    def _patches(self, tp_rank=0, tp_size=2):
        return [
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib"),
        ]

    def _start(self, patches):
        mocks = [p.start() for p in patches]
        pcp_group = MagicMock()
        pcp_group.world_size = 1
        mocks[2].return_value = pcp_group  # get_pcp_group -> pcp_group
        mocks[5].import_module.return_value = MagicMock()  # importlib.import_module
        return mocks

    def _make_worker(
        self,
        *,
        tp_size=2,
        tp_rank=0,
        kv_role="kv_consumer",
        extra_config=None,
        num_kv_heads=8,
        use_sparse=False,
        use_layerwise=False,
    ):
        patches = self._patches(tp_rank=tp_rank, tp_size=tp_size)
        self._start(patches)
        try:
            cfg = self._make_vllm_config(
                kv_role=kv_role, extra_config=extra_config, num_kv_heads=num_kv_heads, use_sparse=use_sparse
            )
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            return KVPoolWorker(cfg, use_layerwise=use_layerwise)
        finally:
            for p in patches:
                p.stop()

    def test_tp_mismatch_detected_decode_tp2_prefill_tp4(self):
        worker = self._make_worker(
            tp_size=2, kv_role="kv_consumer", extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
        )
        self.assertTrue(worker.tp_mismatch)
        self.assertEqual(worker.peer_tp_size, 4)
        self.assertEqual(worker.effective_tp_size, 4)
        self.assertEqual(worker.local_heads_per_rank, 4)
        self.assertEqual(worker.effective_heads_per_rank, 2)
        self.assertEqual(worker.num_sub_keys, 2)

    def test_register_kv_caches_initializes_tp_mismatch_strides(self):
        worker = self._make_worker(
            tp_size=2, kv_role="kv_consumer", extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
        )
        fake_cache = MagicMock()
        fake_cache.shape = [100, 16, 4, 64]
        fake_cache.__getitem__.return_value.numel.return_value = 16 * 4 * 64
        fake_cache.element_size.return_value = 2
        fake_cache.stride.return_value = 16 * 4 * 64
        fake_cache.data_ptr.return_value = 10000
        fake_cache.untyped_storage.return_value.data_ptr.return_value = 10000
        worker._transfer_threads_started = True

        worker.register_kv_caches({"layers.0": (fake_cache, fake_cache)})

        self.assertEqual(worker.per_token_bytes, 512)
        self.assertEqual(worker.sub_size_bytes, 256)

    def test_tp_mismatch_disabled_when_no_config(self):
        # No prefill_tp_size/decode_tp_size -> tp_mismatch False (original behavior)
        worker = self._make_worker(
            tp_size=2, kv_role="kv_consumer", extra_config={"backend": "mooncake"}, num_kv_heads=8
        )
        self.assertFalse(worker.tp_mismatch)
        self.assertEqual(worker.num_sub_keys, 1)
        self.assertEqual(worker.effective_tp_size, 2)

    def test_tp_mismatch_disabled_when_peer_equal(self):
        worker = self._make_worker(
            tp_size=2, kv_role="kv_consumer", extra_config={"backend": "mooncake", "prefill_tp_size": 2}, num_kv_heads=8
        )
        self.assertFalse(worker.tp_mismatch)

    def test_tp_mismatch_disabled_when_use_mla(self):
        patches = self._patches(tp_rank=0, tp_size=2)
        self._start(patches)
        try:
            cfg = self._make_vllm_config(
                kv_role="kv_consumer", extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
            )
            cfg.model_config.use_mla = True
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            worker = KVPoolWorker(cfg, use_layerwise=False)
        finally:
            for p in patches:
                p.stop()
        # use_mla -> num_kv_head forced to 1, can't satisfy >= effective_tp_size
        self.assertFalse(worker.tp_mismatch)

    def test_tp_mismatch_rejects_use_sparse(self):
        patches = self._patches(tp_rank=0, tp_size=2)
        self._start(patches)
        try:
            cfg = self._make_vllm_config(
                kv_role="kv_consumer",
                extra_config={"backend": "mooncake", "prefill_tp_size": 4},
                num_kv_heads=8,
                use_sparse=True,
            )
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            with self.assertRaises(ValueError):
                KVPoolWorker(cfg, use_layerwise=False)
        finally:
            for p in patches:
                p.stop()

    def test_tp_mismatch_rejects_layerwise(self):
        patches = self._patches(tp_rank=0, tp_size=2)
        self._start(patches)
        try:
            cfg = self._make_vllm_config(
                kv_role="kv_consumer", extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
            )
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            with self.assertRaises(ValueError):
                KVPoolWorker(cfg, use_layerwise=True)
        finally:
            for p in patches:
                p.stop()

    def test_make_sub_key_str_rewrites_rank(self):
        worker = self._make_worker(
            tp_rank=1, tp_size=2, extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
        )
        rank = worker.metadata[0].head_or_tp_rank  # = 1 for tp_rank=1

        class FakeKey:
            def to_string(self):
                return f"model@head_or_tp_rank:{rank}@pp_rank:0@k0"

        out = worker._make_sub_key_str(FakeKey(), effective_rank=3)
        self.assertIn("@head_or_tp_rank:3", out)
        self.assertNotIn(f"@head_or_tp_rank:{rank}", out)

    def test_build_strided_addrs_uses_stride(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        # Simulate register_kv_caches outputs (group-0 dict structure).
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [1000]}
        worker.group_block_len = {0: [64]}  # bytes per block
        worker.group_block_stride = {0: [128]}  # padded stride (> block_len)
        worker.sub_size_bytes = 8
        addrs, sizes = worker._build_strided_addrs(block_id=2, token_count=3, sub_idx=1)
        # per_token_bytes = 64 // 4 = 16; block_base = 1000 + 2*128 = 1256
        # sub_idx=1 -> head_offset = 8
        # addrs = [1256+0*16+8, 1256+1*16+8, 1256+2*16+8] = [1264, 1280, 1296]
        self.assertEqual(addrs, [1264, 1280, 1296])
        self.assertEqual(sizes, [8, 8, 8])

    def test_build_tp_mismatch_keys_and_addrs_counts_and_ranks(self):
        worker = self._make_worker(
            tp_rank=1, tp_size=2, extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
        )
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [0]}
        worker.group_block_len = {0: [16]}
        worker.group_block_stride = {0: [16]}
        worker.sub_size_bytes = 2

        class FakeKey:
            def __init__(self, i):
                self.i = i

            def to_string(self):
                return f"m@head_or_tp_rank:{worker.metadata[0].head_or_tp_rank}@pp_rank:0@k{self.i}"

        def fake_process_tokens_with_block_ids(token_len, block_hashes, block_ids, mask_num=0):
            yield 0, 4, FakeKey(0), block_ids[0]
            yield 4, 8, FakeKey(1), block_ids[1]

        worker.token_database = MagicMock()
        worker.token_database.process_tokens_with_block_ids.side_effect = fake_process_tokens_with_block_ids

        keys, addrs, sizes, block_ids = worker._build_tp_mismatch_keys_and_addrs(
            block_hashes=[b"h0", b"h1"], block_ids=[10, 11], token_len=8, mask_num=0
        )
        # 2 chunks * num_sub_keys=2 = 4 keys
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(addrs), 4)
        self.assertEqual(len(sizes), 4)
        self.assertEqual(len(block_ids), 4)
        # tp_rank=1, num_sub_keys=2 -> effective_rank = 1*2 + {0,1} = {2,3}
        self.assertIn("@head_or_tp_rank:2", keys[0])
        self.assertIn("@head_or_tp_rank:3", keys[1])

    def test_build_tp_mismatch_keys_and_addrs_skips_missing_block_ids(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [0]}
        worker.group_block_len = {0: [16]}
        worker.group_block_stride = {0: [16]}
        worker.sub_size_bytes = 2

        class FakeKey:
            def __init__(self, i):
                self.i = i

            def to_string(self):
                return f"m@head_or_tp_rank:{worker.metadata[0].head_or_tp_rank}@pp_rank:0@k{self.i}"

        worker.token_database = MagicMock()
        worker.token_database.process_tokens_with_block_ids.return_value = [
            (4, 8, FakeKey(1), 10),
        ]

        keys, addrs, sizes, block_ids = worker._build_tp_mismatch_keys_and_addrs(
            block_hashes=[b"h0", b"h1"], block_ids=[10], token_len=8, mask_num=0
        )

        self.assertEqual(len(keys), 2)
        self.assertEqual(len(addrs), 2)
        self.assertEqual(len(sizes), 2)
        self.assertEqual(block_ids, [10, 10])
        self.assertIn("@k1", keys[0])

    def test_load_kv_tp_mismatch_calls_backend_get(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [0]}
        worker.group_block_len = {0: [16]}
        worker.group_block_stride = {0: [16]}
        worker.sub_size_bytes = 2
        worker.m_store = MagicMock()
        worker.m_store.get.return_value = [0]  # success

        class FakeKey:
            def to_string(self):
                return f"m@head_or_tp_rank:{worker.metadata[0].head_or_tp_rank}@pp_rank:0@k0"

        worker.token_database = MagicMock()
        worker.token_database.process_tokens_with_block_ids.side_effect = lambda *a, **kw: iter([(0, 4, FakeKey(), 5)])

        worker._load_kv_tp_mismatch(block_hashes=[b"h0"], block_ids=[5], token_len=4, mask_num=0)
        worker.m_store.get.assert_called_once()

    def test_store_kv_tp_mismatch_skips_when_not_stored(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.is_stored_request.return_value = False
        req = ReqMeta(
            req_id="r1", token_len_chunk=4, block_ids_by_group=[[5]], block_hashes=[b"h0"], current_event=None
        )
        worker._store_kv_tp_mismatch(req)
        worker.kv_send_thread.dec_stored_request.assert_not_called()

    def test_store_kv_tp_mismatch_puts_missing_and_decrements(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [0]}
        worker.group_block_len = {0: [16]}
        worker.group_block_stride = {0: [16]}
        worker.sub_size_bytes = 2
        worker.m_store = MagicMock()
        worker.enable_kv_events = False

        class FakeKey:
            def to_string(self):
                return f"m@head_or_tp_rank:{worker.metadata[0].head_or_tp_rank}@pp_rank:0@k0"

        worker.token_database = MagicMock()
        worker.token_database.process_tokens_with_block_ids.side_effect = lambda *a, **kw: iter([(0, 4, FakeKey(), 5)])

        send_thread = MagicMock()
        send_thread.is_stored_request.return_value = True
        # 2 sub-keys: first missing, second present -> only the first is put.
        send_thread.lookup.return_value = [False, True]
        worker.kv_send_thread = send_thread

        req = ReqMeta(
            req_id="r1", token_len_chunk=4, block_ids_by_group=[[5]], block_hashes=[b"h0"], current_event=None
        )
        worker._store_kv_tp_mismatch(req)
        worker.m_store.put.assert_called_once()
        put_keys = worker.m_store.put.call_args.args[0]
        self.assertEqual(len(put_keys), 1)
        send_thread.dec_stored_request.assert_called_once_with("r1")

    def test_store_kv_tp_mismatch_decrements_on_put_exception(self):
        worker = self._make_worker(extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8)
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [0]}
        worker.group_block_len = {0: [16]}
        worker.group_block_stride = {0: [16]}
        worker.sub_size_bytes = 2
        worker.m_store = MagicMock()
        worker.m_store.put.side_effect = RuntimeError("put failed")
        worker.enable_kv_events = False

        class FakeKey:
            def to_string(self):
                return f"m@head_or_tp_rank:{worker.metadata[0].head_or_tp_rank}@pp_rank:0@k0"

        worker.token_database = MagicMock()
        worker.token_database.process_tokens_with_block_ids.side_effect = lambda *a, **kw: iter([(0, 4, FakeKey(), 5)])

        send_thread = MagicMock()
        send_thread.is_stored_request.return_value = True
        send_thread.lookup.return_value = [False, False]
        worker.kv_send_thread = send_thread

        req = ReqMeta(
            req_id="r1", token_len_chunk=4, block_ids_by_group=[[5]], block_hashes=[b"h0"], current_event=None
        )
        with self.assertRaises(RuntimeError):
            worker._store_kv_tp_mismatch(req)
        send_thread.dec_stored_request.assert_called_once_with("r1")

    def test_get_group_tp_size_uses_effective_tp(self):
        worker = self._make_worker(
            tp_size=2, extra_config={"backend": "mooncake", "prefill_tp_size": 4}, num_kv_heads=8
        )
        self.assertEqual(worker.get_group_tp_size(0), 4)  # effective_tp_size under tp_mismatch


if __name__ == "__main__":
    unittest.main()
