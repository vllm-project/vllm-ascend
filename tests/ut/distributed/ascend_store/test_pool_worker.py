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
from unittest.mock import MagicMock, call, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LayerBlockRange,
    LayerTransferTask,
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


class TestKVPoolWorkerThreadSelection(unittest.TestCase):
    def _make_worker(
        self,
        backend_name: str,
        use_layerwise: bool,
        kv_role: str = "kv_both",
        consumer_is_to_put: bool = False,
    ):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import is_block_key_layerwise
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
            MooncakeSessionTracker,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = object.__new__(KVPoolWorker)
        worker._transfer_threads_started = False
        worker.use_layerwise = use_layerwise
        worker.use_block_key_layerwise = is_block_key_layerwise(use_layerwise, backend_name)
        worker.use_memcache_gva_layerwise = worker.use_block_key_layerwise and backend_name == "memcache"
        worker.kv_role = kv_role
        worker.consumer_is_to_put = consumer_is_to_put
        worker.load_async = True
        worker.group_uses_align_state = [False]
        worker.enable_kv_events = False
        worker.m_store = MagicMock()
        worker.token_database = MagicMock()
        worker.block_size = 16
        worker.tp_rank = 0
        worker.tp_size = 1
        worker.dcp_size = 1
        worker.put_step = 1
        worker.my_key_index = 0
        worker.num_ranks_per_layer = 1
        worker.page_size_bytes = 16
        worker.num_layers = 2
        worker.h2d_stagger_us = 0
        worker.layerwise_max_transfer_blocks = 0
        worker.layerwise_max_transfer_bytes = 0
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._mooncake_session_tracker = MooncakeSessionTracker()
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._layer_load_aborted = threading.Event()
        return worker

    def test_block_key_backends_use_layer_threads(self):
        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
        for backend_name, use_layerwise, kv_role, consumer_is_to_put, send_family, recv_family in (
            ("memcache", True, "kv_both", False, "layer", "layer"),
            ("mooncake", True, "kv_both", False, "layer", "layer"),
            ("yuanrong", True, "kv_both", False, "key_layer", "key_layer"),
            ("mooncake", False, "kv_both", False, "whole_key", "whole_key"),
            ("mooncake", True, "kv_consumer", False, None, "layer"),
            ("mooncake", True, "kv_consumer", True, "layer", "layer"),
        ):
            with self.subTest(
                backend=backend_name,
                use_layerwise=use_layerwise,
                kv_role=kv_role,
                consumer_is_to_put=consumer_is_to_put,
            ), patch(
                f"{module}.threading.Event",
                return_value=MagicMock(),
            ), patch(
                f"{module}.torch"
            ), patch(f"{module}.KVCacheStoreLayerSendingThread") as layer_send, patch(
                f"{module}.KVCacheStoreLayerRecvingThread"
            ) as layer_recv, patch(f"{module}.KVCacheStoreKeyLayerSendingThread") as key_layer_send, patch(
                f"{module}.KVCacheStoreKeyLayerRecvingThread"
            ) as key_layer_recv, patch(f"{module}.KVCacheStoreSendingThread") as whole_send, patch(
                f"{module}.KVCacheStoreRecvingThread"
            ) as whole_recv:
                self._make_worker(
                    backend_name,
                    use_layerwise,
                    kv_role,
                    consumer_is_to_put,
                )._start_kv_transfer_threads()

                self.assertEqual(layer_send.called, send_family == "layer")
                self.assertEqual(layer_recv.called, recv_family == "layer")
                self.assertEqual(key_layer_send.called, send_family == "key_layer")
                self.assertEqual(key_layer_recv.called, recv_family == "key_layer")
                self.assertEqual(whole_send.called, send_family == "whole_key")
                self.assertEqual(whole_recv.called, recv_family == "whole_key")


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
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {"backend": "mooncake"}
        config.cache_config.block_size = block_size
        config.kv_events_config = None
        return config

    def test_memcache_block_key_layerwise_rejects_non_tp_topology(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        config = self._make_vllm_config(extra_config={"backend": "memcache"})
        config.parallel_config.pipeline_parallel_size = 2
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        worker = object.__new__(KVPoolWorker)

        with self.assertRaisesRegex(ValueError, r"memcache.*pipeline_parallel_size=2"):
            worker._init_kv_transfer_config(config, {"backend": "memcache"}, True, None)

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

    def test_layer_receiver_shares_worker_invalid_block_state(self):
        worker = self._make_worker(
            kv_role="kv_consumer",
            extra_config={"backend": "memcache"},
        )
        worker.use_layerwise = True
        worker.use_block_key_layerwise = True
        worker.page_size_bytes = 64
        worker.token_database.set_group_buffers(
            {0: [1000, 2000, 3000, 4000]},
            {0: [32, 32, 32, 32]},
        )

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreLayerRecvingThread,
        )

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.torch.npu",
                create=True,
            ),
            patch.object(KVCacheStoreLayerRecvingThread, "start"),
            patch.object(threading.Event, "wait", return_value=True),
        ):
            worker._start_kv_transfer_threads()

        recv_thread = worker.kv_recv_thread
        self.assertIsInstance(recv_thread, KVCacheStoreLayerRecvingThread)
        self.assertIs(recv_thread._invalid_block_ids, worker._invalid_block_ids)
        self.assertIs(recv_thread._invalid_block_ids_lock, worker._invalid_block_ids_lock)

        recv_thread._invalid_block_ids.add(7)
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7})

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
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.torch"):
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

    def test_mooncake_layer_tasks_keep_key_major_batch_kind(self):
        worker = self._make_worker()
        worker.backend_name = "mooncake"
        worker.use_block_key_layerwise = True
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=8,
            save_end_token=0,
            block_ids=[10],
            can_save=True,
            partial_block_index=0,
            load_spec=LoadSpec(0, 8, True, 8),
            load_block_keys=["key-tail"],
        )

        worker._process_save_for_layer_batch([request], 0)
        worker._process_load_for_layer_batch([request], 0)

        self.assertTrue(worker.layer_save_tasks[0][0].use_key_major_ranges)
        self.assertTrue(worker.layer_load_tasks[0][0].use_key_major_ranges)

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
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
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


class TestKVPoolWorkerMooncakeLayerSessions(unittest.TestCase):
    @staticmethod
    def _make_worker():
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
            MooncakeSessionTracker,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = object.__new__(KVPoolWorker)
        worker.backend_name = "mooncake"
        worker.use_block_key_layerwise = True
        worker.kv_role = "kv_both"
        worker.consumer_is_to_put = False
        worker.tp_rank = 0
        worker.put_step = 1
        worker.model_name = "model"
        worker.head_or_tp_rank = 0
        worker.block_size = 16
        worker.page_size_bytes = 64
        worker.num_layers = 2
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._mooncake_session_tracker = MooncakeSessionTracker()
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._load_session_lock = threading.Lock()
        worker._layer_load_aborted = threading.Event()
        worker._current_mooncake_request_ids = set()
        worker._current_mooncake_last_chunk_req_ids = set()
        worker.m_store = MagicMock()
        worker.kv_send_thread = MagicMock()
        return worker

    def test_next_chunk_renews_prefix_and_prior_complete_keys(self):
        worker = self._make_worker()
        worker.m_store.batch_put_start.return_value = [0]
        worker.m_store.batch_get_start.return_value = [0]
        first_chunk = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            save_start_token=16,
            save_end_token=32,
            block_ids=[10, 11],
            block_hashes=[b"\x0a", b"\x0b"],
            can_save=True,
            load_spec=LoadSpec(0, 16, True, 16),
            is_last_chunk=False,
        )

        worker._prepare_mooncake_layerwise_sessions([first_chunk])
        worker._mooncake_session_tracker.commit_put_keys(["model@0b@0"])

        worker.m_store.batch_get_start.reset_mock()
        worker.m_store.batch_get_start.return_value = [0, 0]
        second_chunk = ReqMeta(
            req_id="r1",
            token_len_chunk=48,
            save_start_token=32,
            save_end_token=48,
            block_ids=[10, 11, 12],
            block_hashes=[b"\x0a", b"\x0b", b"\x0c"],
            can_save=False,
            is_last_chunk=False,
        )

        worker._prepare_mooncake_layerwise_sessions([second_chunk])

        worker.m_store.batch_get_start.assert_called_once_with(
            ["model@0a@0", "model@0b@0"]
        )
        self.assertEqual(
            second_chunk.load_block_keys,
            ["model@0a@0", "model@0b@0"],
        )
        self.assertEqual(
            second_chunk.load_keys,
            ["model@0a@0", "model@0b@0"],
        )
        worker.layer_load_tasks = [[], []]
        worker._process_load_for_layer_batch([second_chunk], 0)
        self.assertEqual(len(worker.layer_load_tasks[0]), 1)
        self.assertEqual(
            worker.layer_load_tasks[0][0].block_ranges[0].start_block,
            0,
        )
        self.assertEqual(
            worker.layer_load_tasks[0][0].block_ranges[0].end_block,
            2,
        )

    def test_prepare_sessions_opens_all_gets_before_any_puts(self):
        worker = self._make_worker()
        worker.m_store.batch_get_start.return_value = [0]
        worker.m_store.batch_put_start.side_effect = ([0], [0])
        first = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            save_start_token=16,
            save_end_token=32,
            block_ids=[10, 11],
            block_hashes=[b"\x0a", b"\x0b"],
            can_save=True,
            load_spec=LoadSpec(0, 16, True, 16),
        )
        second = ReqMeta(
            req_id="r2",
            token_len_chunk=32,
            save_start_token=16,
            save_end_token=32,
            block_ids=[20, 21],
            block_hashes=[b"\x0a", b"\x0c"],
            can_save=True,
            load_spec=LoadSpec(0, 16, True, 16),
        )

        worker._prepare_mooncake_layerwise_sessions([first, second])

        session_calls = [
            call[0]
            for call in worker.m_store.method_calls
            if call[0] in {"batch_get_start", "batch_put_start"}
        ]
        self.assertEqual(
            session_calls,
            ["batch_get_start", "batch_put_start", "batch_put_start"],
        )

    def test_shared_started_key_survives_new_put_start_failure_for_request(self):
        worker = self._make_worker()
        worker.m_store.batch_put_start.side_effect = (
            [0],
            RuntimeError("put start failed"),
        )
        worker.m_store.batch_revoke.return_value = [0]
        first = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            save_start_token=0,
            save_end_token=16,
            block_ids=[10],
            block_hashes=[b"\x0a"],
            can_save=True,
        )
        second = ReqMeta(
            req_id="r2",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids=[20, 21],
            block_hashes=[b"\x0a", b"\x0b"],
            can_save=True,
        )

        worker._prepare_mooncake_layerwise_sessions([first, second])
        worker._mooncake_session_tracker.commit_put_keys(["model@0a@0"])

        self.assertEqual(second.save_block_keys, ["model@0a@0", None])
        self.assertEqual(
            worker._mooncake_session_tracker.prepare_load_entries("r2", []),
            [("model@0a@0", 0)],
        )

    def test_mixed_last_chunk_shared_key_closes_after_last_owner(self):
        worker = self._make_worker()
        worker.current_layer = 1
        worker.layer_load_tasks = [[], []]
        worker.layer_load_finished_events = [threading.Event(), threading.Event()]
        worker._submit_ready_layer_loads = MagicMock()
        worker.m_store.batch_get_start.side_effect = ([0], [0])
        first = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[10],
            block_hashes=[b"\x0a"],
            load_spec=LoadSpec(0, 16, True, 16),
            is_last_chunk=True,
        )
        second = ReqMeta(
            req_id="r2",
            token_len_chunk=16,
            block_ids=[20],
            block_hashes=[b"\x0a"],
            load_spec=LoadSpec(0, 16, True, 16),
            is_last_chunk=False,
        )

        worker._prepare_mooncake_layerwise_sessions([first, second])
        worker.wait_for_layer_load()

        worker.m_store.batch_get_end.assert_not_called()

        worker.current_layer = 1
        second.is_last_chunk = True
        worker._prepare_mooncake_layerwise_sessions([second])
        worker.wait_for_layer_load()

        worker.m_store.batch_get_end.assert_called_once_with(["model@0a@0"])

    def test_non_saving_rank_skips_put_start_but_opens_get_session(self):
        worker = self._make_worker()
        worker.tp_rank = 1
        worker.put_step = 4
        worker.m_store.batch_get_start.return_value = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            save_end_token=16,
            block_ids=[10],
            block_hashes=[b"\x0a"],
            can_save=True,
            load_spec=LoadSpec(0, 16, True, 16),
        )

        worker._prepare_mooncake_layerwise_sessions([request])

        worker.m_store.batch_put_start.assert_not_called()
        worker.m_store.batch_get_start.assert_called_once_with(["model@0a@0"])
        self.assertEqual(request.save_block_keys, [])
        self.assertEqual(request.load_block_keys, ["model@0a@0"])

    def test_consumer_to_put_remains_a_layerwise_save_owner(self):
        worker = self._make_worker()
        worker.kv_role = "kv_consumer"
        worker.consumer_is_to_put = True
        worker.m_store.batch_put_start.return_value = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            save_end_token=16,
            block_ids=[10],
            block_hashes=[b"\x0a"],
            can_save=True,
        )

        worker._prepare_mooncake_layerwise_sessions([request])

        worker.m_store.batch_put_start.assert_called_once_with(["model@0a@0"], [128])
        self.assertEqual(request.save_block_keys, ["model@0a@0"])

    def test_partial_only_put_start_failures_keep_key_major_tasks(self):
        for failure in ("negative", "malformed", "exception"):
            with self.subTest(failure=failure):
                worker = self._make_worker()
                worker.layer_save_tasks = [[], []]
                if failure == "negative":
                    worker.m_store.batch_put_start.return_value = [-1]
                elif failure == "malformed":
                    worker.m_store.batch_put_start.return_value = []
                else:
                    worker.m_store.batch_put_start.side_effect = RuntimeError(
                        "put start failed"
                    )
                request = ReqMeta(
                    req_id="r1",
                    token_len_chunk=8,
                    save_end_token=0,
                    block_ids=[10],
                    can_save=True,
                    partial_block_index=0,
                )

                worker._prepare_mooncake_layerwise_sessions([request])
                for layer_id in range(worker.num_layers):
                    worker._process_save_for_layer_batch([request], layer_id)

                self.assertIsNone(request.save_last_block_key)
                expected_pending = (
                    set() if failure == "negative" else {"model@r1_lastblock@0"}
                )
                self.assertEqual(worker._put_started_keys, expected_pending)
                if failure == "negative":
                    worker.kv_send_thread.add_revoke_request.assert_not_called()
                else:
                    worker.kv_send_thread.add_revoke_request.assert_called_once_with(
                        ["model@r1_lastblock@0"]
                    )
                worker.m_store.batch_revoke.assert_not_called()
                self.assertEqual(len(worker.layer_save_tasks), 2)
                for layer_tasks in worker.layer_save_tasks:
                    self.assertEqual(len(layer_tasks), 1)
                    self.assertTrue(layer_tasks[0].use_key_major_ranges)

    def test_two_layer_role_matrix_advances_once_and_closes_get_session(self):
        role_matrix = (
            ("kv_consumer", False, False),
            ("kv_consumer", True, True),
            ("kv_producer", False, True),
            ("kv_both", False, True),
        )
        for kv_role, consumer_is_to_put, saves_layers in role_matrix:
            with self.subTest(
                kv_role=kv_role,
                consumer_is_to_put=consumer_is_to_put,
            ):
                worker = self._make_worker()
                worker.kv_role = kv_role
                worker.consumer_is_to_put = consumer_is_to_put
                worker.current_layer = 0
                worker.layer_load_tasks = [[], []]
                worker.layer_save_tasks = [[], []]
                worker.layer_load_finished_events = [
                    threading.Event(),
                    threading.Event(),
                ]
                worker.layer_save_finished_events = [
                    threading.Event(),
                    threading.Event(),
                ]
                worker.sync_save_events = [MagicMock(), MagicMock()]
                worker.kv_send_thread = MagicMock()
                worker._mooncake_session_tracker.prepare_load_entries(
                    "r1", [("key-1", 0)]
                )
                worker._mooncake_session_tracker.record_get_result(
                    "key-1", {"r1"}, succeeded=True
                )
                worker._current_mooncake_request_ids = {"r1"}
                worker._current_mooncake_last_chunk_req_ids = {"r1"}
                worker.m_store.batch_get_end.return_value = 0
                worker._submit_ready_layer_loads = MagicMock()
                metadata = AscendConnectorMetadata(set(), set())

                for expected_layer in range(2):
                    worker.wait_for_layer_load()
                    if saves_layers:
                        self.assertEqual(worker.current_layer, expected_layer)
                        worker.save_kv_layer(metadata)
                    self.assertEqual(worker.current_layer, expected_layer + 1)

                worker.m_store.batch_get_end.assert_called_once_with(["key-1"])

    def test_prepare_sessions_keeps_key_block_alignment_on_partial_start_failures(self):
        worker = self._make_worker()
        worker.m_store.batch_put_start.return_value = [0, -1]
        worker.m_store.batch_get_start.return_value = [0, -1]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            save_end_token=32,
            block_ids=[10, 11],
            block_hashes=[b"\x0a", b"\x0b"],
            can_save=True,
            load_spec=LoadSpec(0, 32, True, 32),
        )

        worker._prepare_mooncake_layerwise_sessions([request])

        self.assertEqual(request.save_block_keys, ["model@0a@0", None])
        self.assertEqual(request.load_block_keys, ["model@0a@0", None])
        self.assertEqual(request.load_keys, ["model@0a@0"])
        self.assertEqual(worker._put_started_keys, {"model@0a@0"})
        self.assertEqual(worker._invalid_block_ids, {11})
        worker.m_store.batch_put_start.assert_called_once_with(
            ["model@0a@0", "model@0b@0"], [128, 128]
        )
        session_calls = [
            call[0]
            for call in worker.m_store.method_calls
            if call[0] in {"batch_get_start", "batch_put_start"}
        ]
        self.assertEqual(
            session_calls,
            ["batch_get_start", "batch_put_start"],
        )

    def test_prepare_sessions_deduplicates_shared_load_keys_across_requests(self):
        worker = self._make_worker()
        worker.m_store.batch_get_start.return_value = [0, -1, 0]
        first_request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"\x0a", b"\x0b"],
            load_spec=LoadSpec(0, 32, True, 32),
        )
        second_request = ReqMeta(
            req_id="r2",
            token_len_chunk=32,
            block_ids=[20, 21],
            block_hashes=[b"\x0a", b"\x0c"],
            load_spec=LoadSpec(0, 32, True, 32),
        )

        worker._prepare_mooncake_layerwise_sessions([first_request, second_request])

        worker.m_store.batch_get_start.assert_called_once_with(
            ["model@0a@0", "model@0b@0", "model@0c@0"]
        )
        self.assertEqual(first_request.load_block_keys, ["model@0a@0", None])
        self.assertEqual(second_request.load_block_keys, ["model@0a@0", "model@0c@0"])
        self.assertEqual(first_request.load_keys, ["model@0a@0"])
        self.assertEqual(second_request.load_keys, ["model@0a@0", "model@0c@0"])
        self.assertEqual(worker._invalid_block_ids, {11})
        worker._release_mooncake_requests_terminal({"r1"})
        worker.m_store.batch_get_end.assert_not_called()
        worker._release_mooncake_requests_terminal({"r2"})
        worker.m_store.batch_get_end.assert_called_once_with(
            ["model@0a@0", "model@0c@0"]
        )

    def test_prepare_sessions_fans_out_shared_load_key_failures(self):
        worker = self._make_worker()
        worker.m_store.batch_get_start.return_value = [-1]
        first_request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[10],
            block_hashes=[b"\x0a"],
            load_spec=LoadSpec(0, 16, True, 16),
        )
        second_request = ReqMeta(
            req_id="r2",
            token_len_chunk=16,
            block_ids=[20],
            block_hashes=[b"\x0a"],
            load_spec=LoadSpec(0, 16, True, 16),
        )

        worker._prepare_mooncake_layerwise_sessions([first_request, second_request])

        worker.m_store.batch_get_start.assert_called_once_with(["model@0a@0"])
        self.assertEqual(first_request.load_block_keys, [None])
        self.assertEqual(second_request.load_block_keys, [None])
        self.assertEqual(worker._invalid_block_ids, {10, 20})

    def test_request_release_owns_batch_get_end_once(self):
        worker = self._make_worker()
        worker._mooncake_session_tracker.prepare_load_entries(
            "r1", [("key-1", 0), ("key-2", 1)]
        )
        worker._mooncake_session_tracker.record_get_result(
            "key-1", {"r1"}, succeeded=True
        )
        worker._mooncake_session_tracker.record_get_result(
            "key-2", {"r1"}, succeeded=True
        )

        worker._release_mooncake_requests_terminal({"r1"})
        worker._release_mooncake_requests_terminal({"r1"})

        worker.m_store.batch_get_end.assert_called_once_with(["key-1", "key-2"])

    def test_preempted_request_releases_owned_load_keys(self):
        worker = self._make_worker()
        worker.kv_send_thread = None
        worker.kv_recv_thread = None
        worker.load_async = False
        worker._mooncake_session_tracker.prepare_load_entries(
            "r1", [("key-1", 0)]
        )
        worker._mooncake_session_tracker.record_get_result(
            "key-1", {"r1"}, succeeded=True
        )
        metadata = AscendConnectorMetadata(set(), {"r1"})

        worker.get_finished(set(), metadata)

        worker.m_store.batch_get_end.assert_called_once_with(["key-1"])
        self.assertEqual(
            worker._mooncake_session_tracker.prepare_load_entries("r1", []),
            [],
        )

    def test_final_layer_completion_closes_load_sessions(self):
        worker = self._make_worker()
        worker.current_layer = 1
        worker.layer_load_tasks = [[], [MagicMock()]]
        worker.layer_load_finished_events = [MagicMock(), MagicMock()]
        worker.layer_load_finished_events[1].wait.return_value = True
        worker._mooncake_session_tracker.prepare_load_entries(
            "r1", [("key-1", 0)]
        )
        worker._mooncake_session_tracker.record_get_result(
            "key-1", {"r1"}, succeeded=True
        )
        worker._current_mooncake_request_ids = {"r1"}
        worker._current_mooncake_last_chunk_req_ids = {"r1"}
        worker._submit_ready_layer_loads = MagicMock()

        worker.wait_for_layer_load()

        worker.m_store.batch_get_end.assert_called_once_with(["key-1"])

    def test_receiver_abort_closes_load_sessions_after_layer_completion(self):
        worker = self._make_worker()
        worker.current_layer = 0
        worker.layer_load_tasks = [[MagicMock()], []]
        worker.layer_load_finished_events = [MagicMock(), MagicMock()]
        worker.layer_load_finished_events[0].wait.return_value = True
        worker._mooncake_session_tracker.prepare_load_entries(
            "r1", [("key-1", 0)]
        )
        worker._mooncake_session_tracker.record_get_result(
            "key-1", {"r1"}, succeeded=True
        )
        worker._current_mooncake_request_ids = {"r1"}
        worker._layer_load_aborted.set()
        worker._submit_ready_layer_loads = MagicMock()

        worker.wait_for_layer_load()

        worker.m_store.batch_get_end.assert_called_once_with(["key-1"])
        self.assertEqual(
            worker._mooncake_session_tracker.prepare_load_entries("r1", []),
            [("key-1", 0)],
        )

    def test_layer_load_timeout_aborts_and_waits_for_completion_before_get_end(self):
        worker = self._make_worker()
        worker.current_layer = 1
        request = ReqMeta(req_id="r1", token_len_chunk=32, block_ids=[10, 11])
        worker.layer_load_tasks = [
            [],
            [
                LayerTransferTask(
                    layer_id=1,
                    block_ranges=[LayerBlockRange(request=request, start_block=0, end_block=2)],
                )
            ],
        ]
        completion_event = MagicMock()

        wait_count = 0

        def wait_for_completion(timeout=None):
            nonlocal wait_count
            wait_count += 1
            if wait_count == 1:
                worker.m_store.batch_get_end.assert_not_called()
                return False
            self.assertTrue(worker._layer_load_aborted.is_set())
            self.assertEqual(timeout, 10)
            worker.m_store.batch_get_end.assert_not_called()
            return True

        completion_event.wait.side_effect = wait_for_completion
        worker.layer_load_finished_events = [MagicMock(), completion_event]
        worker._mooncake_session_tracker.prepare_load_entries(
            "r1", [("key-1", 0)]
        )
        worker._mooncake_session_tracker.record_get_result(
            "key-1", {"r1"}, succeeded=True
        )
        worker._current_mooncake_request_ids = {"r1"}
        worker._submit_ready_layer_loads = MagicMock()

        worker.wait_for_layer_load()

        self.assertEqual(worker._invalid_block_ids, {10, 11})
        self.assertEqual(completion_event.wait.call_count, 2)
        worker.m_store.batch_get_end.assert_called_once_with(["key-1"])

    def test_mooncake_layer_load_that_never_drains_fails_without_get_end(self):
        worker = self._make_worker()
        worker.current_layer = 0
        request = ReqMeta(req_id="r1", token_len_chunk=16, block_ids=[10])
        worker.layer_load_tasks = [
            [
                LayerTransferTask(
                    layer_id=0,
                    block_ranges=[
                        LayerBlockRange(
                            request=request,
                            start_block=0,
                            end_block=1,
                        )
                    ],
                )
            ],
            [],
        ]
        completion_event = MagicMock()
        completion_event.wait.side_effect = [False, False]
        worker.layer_load_finished_events = [completion_event, MagicMock()]
        worker._mooncake_session_tracker.prepare_load_entries(
            "r1", [("key-1", 0)]
        )
        worker._mooncake_session_tracker.record_get_result(
            "key-1", {"r1"}, succeeded=True
        )
        worker._current_mooncake_request_ids = {"r1"}
        worker._submit_ready_layer_loads = MagicMock()

        with self.assertRaisesRegex(
            TimeoutError,
            "refusing to close the in-flight get session",
        ):
            worker.wait_for_layer_load()

        self.assertEqual(
            completion_event.wait.call_args_list,
            [call(timeout=10), call(timeout=10)],
        )
        worker.m_store.batch_get_end.assert_not_called()
        self.assertEqual(
            worker._mooncake_session_tracker.prepare_load_entries("r1", []),
            [("key-1", 0)],
        )

    def test_memcache_layer_load_keeps_unbounded_drain_barrier(self):
        worker = self._make_worker()
        worker.backend_name = "memcache"
        worker.current_layer = 0
        request = ReqMeta(req_id="r1", token_len_chunk=16, block_ids=[10])
        worker.layer_load_tasks = [
            [
                LayerTransferTask(
                    layer_id=0,
                    block_ranges=[
                        LayerBlockRange(
                            request=request,
                            start_block=0,
                            end_block=1,
                        )
                    ],
                )
            ],
            [],
        ]
        completion_event = MagicMock()
        completion_event.wait.side_effect = [False, True]
        worker.layer_load_finished_events = [completion_event, MagicMock()]
        worker._submit_ready_layer_loads = MagicMock()

        worker.wait_for_layer_load()

        self.assertEqual(
            completion_event.wait.call_args_list,
            [call(timeout=10), call()],
        )
        worker.m_store.batch_get_end.assert_not_called()

    def test_prepare_sessions_uses_last_block_key_at_hashless_boundary(self):
        worker = self._make_worker()
        worker.m_store.batch_get_start.return_value = [0, 0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"\x0a"],
            load_spec=LoadSpec(0, 32, True, 32),
        )

        worker._prepare_mooncake_layerwise_sessions([request])

        self.assertEqual(
            request.load_block_keys,
            ["model@0a@0", "model@r1_lastblock@0"],
        )
        self.assertIsNone(request.load_last_block_key)
        self.assertEqual(request.load_keys, ["model@0a@0", "model@r1_lastblock@0"])

    def test_put_start_shape_error_queues_revoke_and_tracks_pending_keys(self):
        worker = self._make_worker()
        worker.m_store.batch_put_start.return_value = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            save_end_token=32,
            block_ids=[10, 11],
            block_hashes=[b"\x0a", b"\x0b"],
            can_save=True,
        )

        worker._prepare_mooncake_layerwise_sessions([request])

        self.assertEqual(request.save_block_keys, [None, None])
        self.assertEqual(worker._put_started_keys, {"model@0a@0", "model@0b@0"})
        worker.kv_send_thread.add_revoke_request.assert_called_once_with(
            ["model@0a@0", "model@0b@0"]
        )
        worker.m_store.batch_revoke.assert_not_called()

    def test_get_start_shape_error_ends_all_keys_and_marks_all_blocks_invalid(self):
        worker = self._make_worker()
        worker.m_store.batch_get_start.return_value = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"\x0a", b"\x0b"],
            load_spec=LoadSpec(0, 32, True, 32),
        )

        worker._prepare_mooncake_layerwise_sessions([request])

        self.assertEqual(request.load_block_keys, [None, None])
        self.assertEqual(worker._invalid_block_ids, {10, 11})
        worker.m_store.batch_get_end.assert_called_once_with(["model@0a@0", "model@0b@0"])

    def test_get_start_shape_error_preserves_unrelated_shared_key_owner(self):
        failures = {
            "exception": RuntimeError("get start failed"),
            "short": [],
            "long": [0, 0],
            "non_integer": [0.5],
        }
        for failure, result in failures.items():
            with self.subTest(failure=failure):
                worker = self._make_worker()
                shared_key = "model@0a@0"
                worker._mooncake_session_tracker.prepare_load_entries(
                    "old-owner",
                    [(shared_key, 0)],
                )
                worker._mooncake_session_tracker.record_get_result(
                    shared_key,
                    {"old-owner", "new-owner"},
                    succeeded=True,
                )
                if isinstance(result, Exception):
                    worker.m_store.batch_get_start.side_effect = result
                else:
                    worker.m_store.batch_get_start.return_value = result
                request = ReqMeta(
                    req_id="new-owner",
                    token_len_chunk=16,
                    block_ids=[10],
                    block_hashes=[b"\x0a"],
                    load_spec=LoadSpec(0, 16, True, 16),
                )

                worker._prepare_mooncake_layerwise_sessions([request])

                self.assertEqual(request.load_block_keys, [None])
                self.assertEqual(worker._invalid_block_ids, {10})
                worker.m_store.batch_get_end.assert_not_called()

                worker._release_mooncake_requests_terminal({"old-owner"})
                worker.m_store.batch_get_end.assert_called_once_with([shared_key])
                self.assertEqual(
                    worker._mooncake_session_tracker.prepare_load_entries(
                        "new-owner",
                        [],
                    ),
                    [(shared_key, 0)],
                )


if __name__ == "__main__":
    unittest.main()
