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

import unittest
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
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
        worker.hash_block_size = 128
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
        worker.m_store.register_buffer.assert_called_once()

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

    def test_process_save_for_layer_batch_skip_no_save(self):
        worker = self._make_worker()
        req = ReqMeta(req_id="r1", token_len_chunk=32, block_ids=[0, 1], block_hashes=["h0", "h1"], can_save=False)
        worker._process_save_for_layer_batch([req], 0)
        self.assertEqual(len(worker.layer_save_tasks[0]), 0)

    def test_process_save_for_layer_batch_skip_zero_range(self):
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
        worker._process_save_for_layer_batch([req], 0)
        self.assertEqual(len(worker.layer_save_tasks[0]), 0)

    def test_process_load_for_layer_batch_skip_no_load(self):
        worker = self._make_worker()
        req = ReqMeta(req_id="r1", token_len_chunk=32, block_ids=[0, 1], block_hashes=["h0", "h1"], load_spec=None)
        worker._process_load_for_layer_batch([req], 0)
        self.assertEqual(len(worker.layer_load_tasks[0]), 0)

    def test_process_load_for_layer_batch_skip_cannot_load(self):
        worker = self._make_worker()
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=0, can_load=False, token_len=0)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=load_spec,
        )
        worker._process_load_for_layer_batch([req], 0)
        self.assertEqual(len(worker.layer_load_tasks[0]), 0)


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
