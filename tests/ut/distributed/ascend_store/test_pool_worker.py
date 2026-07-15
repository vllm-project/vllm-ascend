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


class FakeStorage:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class FakeTensorBlock:
    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel


class FakeTensor:
    def __init__(
        self,
        data_ptr,
        shape=(4, 2, 4),
        element_size=2,
        stride0=8,
        storage_ptr=None,
    ):
        self._data_ptr = data_ptr
        self.shape = list(shape)
        self._element_size = element_size
        self._stride0 = stride0
        self._storage_ptr = data_ptr if storage_ptr is None else storage_ptr

    def data_ptr(self):
        return self._data_ptr

    def element_size(self):
        return self._element_size

    def stride(self, dim):
        if dim != 0:
            raise ValueError("FakeTensor only supports stride(0)")
        return self._stride0

    def untyped_storage(self):
        return FakeStorage(self._storage_ptr)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        block_numel = 1
        for size in self.shape[1:]:
            block_numel *= size
        return FakeTensorBlock(block_numel)


class CountingIterator:
    def __init__(self, values):
        self.values = list(values)
        self.next_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        if not self.values:
            raise StopIteration
        return self.values.pop(0)


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

    def test_find_max_hit_index_found(self):
        cls = self._make_worker_class()
        arr = [[1, 1, 0], [1, 0, 1]]
        result = cls.find_max_hit_index(None, arr, 3)
        self.assertEqual(result, 0)

    def test_find_max_hit_index_all_one(self):
        cls = self._make_worker_class()
        arr = [[1, 1, 1], [1, 1, 1]]
        result = cls.find_max_hit_index(None, arr, 3)
        self.assertEqual(result, 2)

    def test_find_max_hit_index_first_pos(self):
        cls = self._make_worker_class()
        arr = [[0, 1], [1, 0]]
        result = cls.find_max_hit_index(None, arr, 3)
        self.assertEqual(result, -1)

    def test_find_max_hit_index_empty(self):
        cls = self._make_worker_class()
        result = cls.find_max_hit_index(None, [], 0)
        self.assertEqual(result, -1)

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

        worker = KVPoolWorker(config, use_layerwize=False)

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

        worker = KVPoolWorker(config, use_layerwize=False)
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

        worker = KVPoolWorker(config, use_layerwize=False)
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

        worker = KVPoolWorker(config, use_layerwize=False)
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

        worker = KVPoolWorker(config, use_layerwize=False)
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

        worker = KVPoolWorker(config, use_layerwize=False)
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

        worker = KVPoolWorker(config, use_layerwize=False)
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

        worker = KVPoolWorker(config, use_layerwize=False)
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
    def test_get_and_clear_finished_requests(
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

        worker = KVPoolWorker(config, use_layerwize=False)
        self.assertIsNotNone(worker.token_database.partitions)
        self.assertEqual(worker.token_database.partitions, [16, 16])


class TestKVPoolWorkerRegisterAndTransfer(unittest.TestCase):
    """Test register_kv_caches, start_load_kv, wait_for_save, get_finished, lookup_scheduler."""

    def _patch_all(self, tp_rank=0, tp_size=1):
        """Return a dict of started patches."""
        patches = {
            "tp_rank": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            "tp_size": patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
                return_value=tp_size,
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

    def _make_worker(self, kv_role="kv_producer", extra_config=None, tp_rank=0, tp_size=1, use_layerwise=False):
        self._patch_all(tp_rank=tp_rank, tp_size=tp_size)
        config = self._make_config(kv_role=kv_role, extra_config=extra_config)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwize=use_layerwise)
        return worker

    def _set_default_group_buffers(self, worker, groups=1):
        group_addrs = {group_id: [1000 + group_id * 1000, 2000 + group_id * 1000] for group_id in range(groups)}
        group_block_len = {group_id: [160, 320] for group_id in range(groups)}
        worker.token_database.set_group_buffers(group_addrs, group_block_len)
        return group_addrs, group_block_len

    def _make_load_meta(
        self,
        req_id="r1",
        token_len=32,
        block_ids=None,
        block_ids_by_group=None,
        kv_cache_group_ids=None,
        load_spec=None,
    ):
        block_count = (token_len + 15) // 16
        if load_spec is None:
            load_spec = LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=token_len,
                can_load=True,
                token_len=token_len,
            )
        return ReqMeta(
            req_id=req_id,
            token_len_chunk=token_len,
            block_ids=block_ids or list(range(block_count)),
            block_ids_by_group=block_ids_by_group,
            block_hashes=[f"h{i}" for i in range(block_count)],
            load_spec=load_spec,
            kv_cache_group_ids=kv_cache_group_ids,
        )

    def _make_connector_meta(self, *requests):
        meta = AscendConnectorMetadata(set(), set())
        for request in requests:
            meta.add_request(request)
        return meta

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
        worker.register_kv_caches(kv_caches)
        self.assertEqual(len(worker.group_kv_caches_base_addr[0]), 2)
        worker.m_store.register_buffer.assert_called_once()

    def test_register_kv_caches_registers_merged_storage_regions(self):
        worker = self._make_worker()
        first_cache = FakeTensor(1000, storage_ptr=1000)
        second_cache = FakeTensor(1100, storage_ptr=1000)

        worker.register_kv_caches({"layer.0": (first_cache, second_cache)})

        worker.m_store.register_buffer.assert_called_once_with([1000], [164])

    def test_register_kv_caches_registers_distinct_storage_regions(self):
        worker = self._make_worker()
        first_cache = FakeTensor(1000, storage_ptr=1000)
        second_cache = FakeTensor(2000, storage_ptr=2000)

        worker.register_kv_caches({"layer.0": (first_cache, second_cache)})

        worker.m_store.register_buffer.assert_called_once_with([1000, 2000], [64, 64])

    def test_register_kv_caches_sets_token_database_group_buffers(self):
        worker = self._make_worker()
        worker.token_database.set_group_buffers = MagicMock()
        kv_caches = {
            "layer.0": (FakeTensor(1000), FakeTensor(2000)),
            "layer.1": (FakeTensor(3000), FakeTensor(4000)),
        }

        worker.register_kv_caches(kv_caches)

        worker.token_database.set_group_buffers.assert_called_once_with(
            {0: [1000, 2000, 3000, 4000]},
            {0: [16, 16, 16, 16]},
            {0: [16, 16, 16, 16]},
            cache_role="kv",
            group_cache_families={0: "default"},
            group_num_layers={0: 2},
        )

    def test_register_kv_caches_sets_group_num_layers(self):
        worker = self._make_worker()
        kv_caches = {
            "layer.0": (FakeTensor(1000), FakeTensor(2000)),
            "layer.1": (FakeTensor(3000), FakeTensor(4000)),
            "layer.2": (FakeTensor(5000), FakeTensor(6000)),
        }

        worker.register_kv_caches(kv_caches)

        self.assertEqual(worker.group_num_layers, {0: 3})

    def test_sparse_metadata_uses_single_kv_head(self):
        self._patch_all(tp_rank=0, tp_size=4)
        config = self._make_config()
        config.model_config.hf_text_config = MagicMock()
        config.model_config.hf_text_config.index_topk = 8
        config.model_config.get_total_num_kv_heads.return_value = 16
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwize=False)

        self.assertTrue(worker.use_sparse)
        self.assertEqual(worker._get_group_num_kv_heads(0), 1)
        self.assertEqual(worker.get_group_tp_size(0), 1)

    def test_register_kv_caches_layerwise_producer_creates_send_and_recv_threads(self):
        def make_recv_thread(*args):
            thread = MagicMock()
            ready_event = args[5]
            thread.start.side_effect = ready_event.set
            return thread

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreLayerSendingThread"
            ) as mock_send_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreLayerRecvingThread",
                side_effect=make_recv_thread,
            ) as mock_recv_cls,
        ):
            worker = self._make_worker(use_layerwise=True)
            worker.register_kv_caches({"layer.0": (FakeTensor(1000), FakeTensor(2000))})

        mock_send_cls.assert_called_once()
        mock_recv_cls.assert_called_once()
        worker.kv_send_thread.start.assert_called_once()
        worker.kv_recv_thread.start.assert_called_once()

    def test_register_kv_caches_non_layerwise_load_async_false_no_recv_thread(self):
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreSendingThread"
            ) as mock_send_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreRecvingThread"
            ) as mock_recv_cls,
        ):
            worker = self._make_worker(extra_config={"backend": "mooncake", "load_async": False})
            worker.register_kv_caches({"layer.0": (FakeTensor(1000), FakeTensor(2000))})

        mock_send_cls.assert_called_once()
        mock_recv_cls.assert_not_called()

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

    def test_start_load_kv_sync_rotates_for_nonzero_tp_rank(self):
        worker = self._make_worker(tp_rank=1, tp_size=2)
        self._set_default_group_buffers(worker)
        worker.m_store.get.return_value = [0, 0, 0]
        req = self._make_load_meta(token_len=48, block_ids=[10, 11, 12])

        worker.start_load_kv(self._make_connector_meta(req))

        keys, addrs, sizes = worker.m_store.get.call_args.args
        self.assertEqual([key.rsplit("@", 1)[-1] for key in keys], ["h1", "h2", "h0"])
        self.assertEqual(addrs[0], [1000 + 11 * 160, 2000 + 11 * 320])
        self.assertEqual(addrs[1], [1000 + 12 * 160, 2000 + 12 * 320])
        self.assertEqual(sizes, [[160, 320], [160, 320], [160, 320]])

    def test_start_load_kv_sync_records_failed_blocks(self):
        worker = self._make_worker()
        self._set_default_group_buffers(worker)
        worker.m_store.get.return_value = [0, 1, 0]
        req = self._make_load_meta(token_len=48, block_ids=[10, 11, 12])

        worker.start_load_kv(self._make_connector_meta(req))

        self.assertEqual(worker.get_block_ids_with_load_errors(), {11})

    def test_start_load_kv_sync_get_none_records_blocks(self):
        worker = self._make_worker()
        self._set_default_group_buffers(worker)
        worker.m_store.get.return_value = None
        req = self._make_load_meta(token_len=48, block_ids=[10, 11, 12])

        worker.start_load_kv(self._make_connector_meta(req))

        self.assertEqual(worker.get_block_ids_with_load_errors(), {10, 11, 12})

    def test_start_load_kv_sync_hybrid_failure_does_not_update_invalid_blocks(self):
        worker = self._make_worker()
        self._set_default_group_buffers(worker, groups=2)
        worker.grouped_block_size = [16, 16]
        worker.group_uses_align_state = [False, False]
        worker.token_database.metadata.append(worker.token_database.metadata[0])
        worker.m_store.get.return_value = [1, 1, 1, 1]
        req = self._make_load_meta(
            token_len=32,
            block_ids_by_group=[[10, 11], [20, 21]],
            kv_cache_group_ids=[0, 1],
        )

        worker.start_load_kv(self._make_connector_meta(req))

        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_start_load_kv_adjusts_token_len_for_last_partial_chunk(self):
        worker = self._make_worker()
        self._set_default_group_buffers(worker)
        worker.m_store.get.return_value = [0, 0]
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=31, can_load=True, token_len=0)
        req = self._make_load_meta(token_len=32, block_ids=[10, 11], load_spec=load_spec)

        worker.start_load_kv(self._make_connector_meta(req))

        self.assertEqual(req.load_spec.token_len, 32)
        self.assertEqual(len(worker.m_store.get.call_args.args[0]), 2)

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

    def test_wait_for_layer_load_final_layer_consumes_ret_mask(self):
        worker = self._make_worker()
        worker.current_layer = worker.num_layers - 1
        ret_sum = MagicMock()
        ret_sum.item.return_value = 7
        ret_mask = MagicMock()
        ret_mask.sum.return_value = ret_sum
        worker.layerwise_retrievers = [iter([ret_mask])]

        worker.wait_for_layer_load()

        ret_mask.sum.assert_called_once()
        ret_sum.item.assert_called_once()

    def test_save_kv_layer_initializes_storers_on_first_layer(self):
        worker = self._make_worker()
        worker.kv_send_thread = MagicMock()
        storer = CountingIterator([None])
        worker.store_layer = MagicMock(return_value=storer)
        event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=["h0"],
            can_save=True,
        )
        meta = self._make_connector_meta(req)

        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.torch.npu.Event") as event_cls:
            event_cls.return_value = event
            worker.save_kv_layer(meta)

        event.record.assert_called_once()
        worker.kv_send_thread.add_stored_request.assert_called_once_with("r1")
        worker.store_layer.assert_called_once_with(req, event)
        self.assertEqual(storer.next_calls, 1)
        self.assertEqual(worker.current_layer, 1)

    def test_save_kv_layer_advances_all_storers(self):
        worker = self._make_worker()
        worker.current_layer = 1
        first_storer = CountingIterator([None])
        second_storer = CountingIterator([None])
        worker.layerwise_storers = [first_storer, second_storer]

        worker.save_kv_layer(self._make_connector_meta())

        self.assertEqual(first_storer.next_calls, 1)
        self.assertEqual(second_storer.next_calls, 1)
        self.assertEqual(worker.current_layer, 2)

    def test_retrieve_layer_yields_each_layer_and_final_mask(self):
        worker = self._make_worker()
        worker.get_event = MagicMock()
        worker.get_event.wait.return_value = True
        worker.kv_recv_thread = MagicMock()
        req = self._make_load_meta(token_len=32, block_ids=[0, 1])

        retriever = worker.retrieve_layer(req)
        self.assertIsNone(next(retriever))
        self.assertIsNone(next(retriever))
        ret_mask = next(retriever)

        self.assertIsNotNone(ret_mask)
        self.assertEqual(worker.kv_recv_thread.add_request.call_count, 2)
        layer_ids = [call.args[0].layer_id for call in worker.kv_recv_thread.add_request.call_args_list]
        self.assertEqual(layer_ids, [0, 1])

    def test_store_layer_yields_layer_metadata_with_hashes(self):
        worker = self._make_worker()
        worker.kv_send_thread = MagicMock()
        current_event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            can_save=True,
            token_ids=list(range(32)),
            original_block_size=16,
            is_last_chunk=True,
        )

        storer = worker.store_layer(req, current_event)
        next(storer)

        layer_meta = worker.kv_send_thread.add_request.call_args.args[0]
        self.assertEqual(layer_meta.req_id, "r1")
        self.assertEqual(layer_meta.layer_id, 0)
        self.assertEqual(layer_meta.starts, [0, 16])
        self.assertEqual(layer_meta.ends, [16, 32])
        self.assertEqual(layer_meta.block_ids, [0, 1])
        self.assertEqual(layer_meta.token_ids, list(range(32)))
        self.assertEqual(layer_meta.block_hashes, ["h0", "h1"])
        self.assertIs(layer_meta.current_event, current_event)

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

    def test_lookup_scheduler_gate_filters_non_c1_groups(self):
        worker = self._make_worker()
        worker.kv_cache_group_families = ["c1", "c4", "default"]
        worker.group_uses_align_state = [False, False, False]
        worker.grouped_block_size = [16, 16, 16]
        worker.num_kv_cache_groups = 3
        worker.token_database.metadata.extend([worker.token_database.metadata[0], worker.token_database.metadata[0]])
        worker.m_store.exists.return_value = [1, 1]

        result = worker.lookup_scheduler(
            32,
            ["h0", "h1"],
            kv_cache_group_ids=[0, 1, 2],
            use_layerwise=False,
        )

        self.assertEqual(result, 32)
        worker.m_store.exists.assert_called_once()
        keys = worker.m_store.exists.call_args.args[0]
        self.assertEqual(len(keys), 2)
        self.assertTrue(all("@group:0" in key for key in keys))

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
        worker.get_and_clear_finished_requests(set(), meta)
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
        use_layerwize=False,
    ):
        patches = self._patches(tp_rank=tp_rank, tp_size=tp_size)
        self._start(patches)
        try:
            cfg = self._make_vllm_config(
                kv_role=kv_role, extra_config=extra_config, num_kv_heads=num_kv_heads, use_sparse=use_sparse
            )
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            return KVPoolWorker(cfg, use_layerwize=use_layerwize)
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

            worker = KVPoolWorker(cfg, use_layerwize=False)
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
                KVPoolWorker(cfg, use_layerwize=False)
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
                KVPoolWorker(cfg, use_layerwize=True)
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
