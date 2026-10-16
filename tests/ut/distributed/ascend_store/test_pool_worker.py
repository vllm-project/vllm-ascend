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
import torch
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    HybridCacheC128Config,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
    TransferChunkWithBlockId,
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

    @staticmethod
    def _c128_chunk(value_start: int, value_end: int, block_id: int = 1):
        return TransferChunkWithBlockId(
            raw_start=value_start * 128,
            raw_end=value_end * 128,
            value_start=value_start,
            value_end=value_end,
            target_block_index=0,
            key=PoolKey(KeyMetadata("hybrid-model", 0, 0, 0, 0), "hash"),
            block_id=block_id,
        )

    def test_merge_c128_chunks_aggregates_authoritative_ranges(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.hybrid_cache_c128_config = HybridCacheC128Config(enabled=True, c128_slots_per_page=128)
        target = torch.full((2, 128, 2), -1, dtype=torch.int64)
        staging = torch.arange(256, dtype=torch.int64).view(1, 128, 2)
        worker.group_kv_cache_tensors = {1: [target]}
        worker.group_block_size_scales = {1: [1]}
        worker.c128_staging_tensors = {1: [staging]}

        worker._merge_c128_staging_chunk(1, self._c128_chunk(0, 4))
        first_range = target[1, 0:4].clone()
        staging.add_(1000)
        worker._merge_c128_staging_chunk(1, self._c128_chunk(4, 8))

        self.assertTrue(torch.equal(first_range, torch.arange(8).view(4, 2)))
        self.assertTrue(torch.equal(target[1, 0:4], first_range))
        self.assertTrue(torch.equal(target[1, 4:8], staging[0, 4:8]))
        self.assertTrue(torch.all(target[1, 8:] == -1))
        self.assertTrue(torch.all(target[0] == -1))

    def test_merge_c128_chunk_rejects_invalid_range(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.hybrid_cache_c128_config = HybridCacheC128Config(enabled=True, c128_slots_per_page=128)
        worker.group_kv_cache_tensors = {1: [torch.zeros((1, 128))]}
        worker.group_block_size_scales = {1: [1]}
        worker.c128_staging_tensors = {1: [torch.zeros((1, 128))]}

        with self.assertRaises(ValueError):
            worker._merge_c128_staging_chunk(1, self._c128_chunk(124, 132, block_id=0))

    def test_external_coordinator_lookup_disables_eagle_drop(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.hybrid_cache_c128_config = HybridCacheC128Config(enabled=True, c128_slots_per_page=128)
        worker.num_kv_cache_groups = 1
        worker.cache_coordinator = MagicMock()
        worker.cache_coordinator.find_longest_cache_hit.return_value = ((), 128)
        worker.m_store = MagicMock()
        worker.m_store.exists.return_value = [1]

        key = MagicMock()
        key.chunk_hash = "ab" * 32
        key.to_string.return_value = "key"
        worker.token_database = MagicMock()
        worker.token_database.process_transfer_chunks.return_value = [MagicMock(key=key)]

        with self.assertLogs(level="INFO") as logs:
            hit = worker._lookup_with_coordinator(
                128,
                [b"h0"],
                [0],
                use_layerwise=False,
                include_all_ranks=False,
            )
        self.assertTrue(any("kv Test lookup" in line for line in logs.output))

        self.assertEqual(hit, 128)
        worker.cache_coordinator.find_longest_cache_hit.assert_called_once()
        self.assertFalse(worker.cache_coordinator.find_longest_cache_hit.call_args.kwargs["apply_eagle"])

    def test_range_save_shards_replicated_kv_heads_across_tp(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.dcp_size = 1
        worker.tp_rank = 2
        worker.put_step = 8
        worker.group_uses_align_state = [False]
        request = ReqMeta("request", 128)
        records = list(range(16))

        selected = worker._shard_range_save_records(request, 0, records)

        self.assertEqual(selected, [2, 10])

    def test_range_save_respects_disable_tp_key_sharding(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.dcp_size = 1
        worker.tp_rank = 2
        worker.put_step = 8
        worker.group_uses_align_state = [False]
        request = ReqMeta("request", 128, disable_tp_key_sharding=True)
        records = list(range(4))

        selected = worker._shard_range_save_records(request, 0, records)

        self.assertEqual(selected, records)

    def test_range_chunk_history_requires_all_store_objects(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker._normalize_range_request_identity = MagicMock()
        worker.group_uses_align_state = [False]
        record = MagicMock()
        record.key = "missing-key"
        worker._range_records = MagicMock(return_value=[record])
        worker._get_range_lease = MagicMock()
        worker.m_store = MagicMock()
        worker.m_store.exists.return_value = [0]
        worker._range_read_plans = {}
        request = ReqMeta(
            "request",
            128,
            block_ids=[0],
            kv_cache_group_ids=[0],
        )

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_session import (
            StoreCommitError,
        )

        with self.assertRaises(StoreCommitError):
            worker._start_load_range_request(
                request,
                128,
                require_all=True,
            )

    def test_range_group_load_rotates_keys_and_destinations_by_tp_rank(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.tp_rank = 2
        worker._range_load_layers_seen = set()
        worker._range_components = {
            (0, "layer.0"): MagicMock(),
            (0, "layer.1"): MagicMock(),
            (0, "layer.2"): MagicMock(),
            (0, "layer.3"): MagicMock(),
        }
        worker._range_destination_for_layers = MagicMock(
            return_value=(
                ["key-0", "key-1", "key-2"],
                [[100], [200], [300]],
                [[10], [20], [30]],
                [[1], [2], [3]],
            )
        )
        request = MagicMock()
        lease = MagicMock()
        lease.load_layer.return_value = [30, 10, 20]
        records = [MagicMock()]
        worker._range_read_plans = {
            "request": (request, lease, records)
        }

        worker._wait_for_layer_load_ranges(
            ("layer.0", "layer.1", "layer.2", "layer.3")
        )

        worker._range_destination_for_layers.assert_called_once_with(
            ("layer.0", "layer.1", "layer.2", "layer.3"), records
        )
        lease.load_layer.assert_called_once_with(
            ["key-2", "key-0", "key-1"],
            [[300], [100], [200]],
            [[30], [10], [20]],
            [[3], [1], [2]],
        )
        self.assertEqual(worker._range_read_plans, {})

    def test_range_destination_flattens_layers_per_store_key(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)

        def component(
            base_addr: int,
            block_len: int,
            block_stride: int,
            object_offset: int,
        ):
            value = MagicMock()
            value.base_addr = base_addr
            value.block_len = block_len
            value.block_stride = block_stride
            value.object_offset = object_offset
            return value

        record = MagicMock()
        record.key = "key"
        record.group_id = 0
        record.block_id = 2
        worker._range_components = {
            (0, "layer.0"): [component(100, 4, 10, 0)],
            (0, "layer.1"): [
                component(200, 5, 20, 64),
                component(300, 6, 30, 128),
            ],
            (0, "layer.2"): [component(400, 7, 40, 192)],
            (0, "layer.3"): [component(500, 8, 50, 256)],
        }

        keys, ptrs, sizes, offsets = worker._range_destination_for_layers(
            ("layer.0", "layer.1", "layer.2", "layer.3"),
            [record],
        )

        self.assertEqual(keys, ["key"])
        self.assertEqual(ptrs, [[120, 240, 360, 480, 600]])
        self.assertEqual(sizes, [[4, 5, 6, 7, 8]])
        self.assertEqual(offsets, [[0, 64, 128, 192, 256]])

    def test_range_request_generation_uses_bounded_global_counter(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker._range_request_generations = {}
        worker._next_range_request_generation = 0

        first = ReqMeta("reused", 128)
        worker._normalize_range_request_identity(first)
        worker._drop_range_request_identity(first.req_id)
        second = ReqMeta("reused", 128)
        worker._normalize_range_request_identity(second)
        worker._drop_range_request_identity(second.req_id)

        self.assertEqual(first.request_generation, 1)
        self.assertEqual(second.request_generation, 2)
        self.assertEqual(worker._range_request_generations, {})
        self.assertEqual(worker._next_range_request_generation, 2)
        self.assertFalse(hasattr(worker, "_range_generation_counters"))


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

        worker = KVPoolWorker(config, use_layerwize=False)
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
        worker.register_kv_caches(kv_caches)
        self.assertEqual(len(worker.group_kv_caches_base_addr[0]), 2)
        worker.m_store.register_buffer.assert_called_once()
        worker.m_store.register_additional_buffer.assert_not_called()

    def test_register_kv_caches_registers_c128_staging_separately(self):
        worker = self._make_worker()
        worker.hybrid_cache_c128_config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace="hybrid_c128_chunk_v1",
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        fake_cache = torch.zeros((2, 128), dtype=torch.uint8)

        worker.register_kv_caches({"layer.0": fake_cache})

        worker.m_store.register_buffer.assert_called_once()
        staging = worker.c128_staging_tensors[0][0]
        worker.m_store.register_additional_buffer.assert_called_once_with(
            [staging.data_ptr()],
            [staging.numel() * staging.element_size()],
        )

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

    def _make_c128_load_worker(self, get_result):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = object.__new__(KVPoolWorker)
        target = torch.full((2, 128, 2), -1, dtype=torch.int64)
        staging = torch.zeros((1, 128, 2), dtype=torch.int64)
        chunk = TransferChunkWithBlockId(
            raw_start=0,
            raw_end=512,
            value_start=0,
            value_end=4,
            target_block_index=0,
            key=PoolKey(KeyMetadata("hybrid-model", 0, 0, 0, 0), "hash"),
            block_id=1,
        )
        worker.hybrid_cache_c128_config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace="hybrid_c128_chunk_v1",
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        worker.cache_transfer_granularity = 512
        worker.use_layerwise = False
        worker.load_async = False
        worker.group_uses_align_state = [False, False]
        worker.tp_rank = 0
        worker._invalid_block_ids = set()
        worker.grouped_block_size = [512, 128]
        worker.group_kv_cache_tensors = {1: [target]}
        worker.group_block_size_scales = {1: [1]}
        worker.c128_staging_tensors = {1: [staging]}
        worker.token_database = MagicMock()
        worker.token_database.load_mask.return_value = None
        worker.token_database.mask_allows_chunk.return_value = True
        worker.token_database.is_c128_group.return_value = True
        worker.token_database.process_transfer_chunks_with_block_ids.return_value = [chunk]
        worker.token_database.prepare_transfer_value.side_effect = lambda _chunk, _block_ids, **_kwargs: (
            [target[1].data_ptr()],
            [target[1].numel() * target[1].element_size()],
            1,
        )
        worker.m_store = MagicMock()

        def get(*_args):
            if get_result == [0]:
                staging.copy_(torch.arange(256, dtype=torch.int64).view(1, 128, 2))
            return get_result

        worker.m_store.get.side_effect = get
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=512,
            block_ids_by_group=[[], [1]],
            block_hashes=["h0", "h1", "h2", "h3"],
            load_spec=LoadSpec(0, 512, True),
            kv_cache_group_ids=[1],
        )
        metadata = AscendConnectorMetadata(set(), set())
        metadata.add_request(request)
        return worker, target, metadata

    def test_start_load_kv_c128_loads_page_directly(self):
        worker, target, metadata = self._make_c128_load_worker([0])

        with self.assertLogs(level="INFO") as logs:
            worker.start_load_kv(metadata)
        self.assertTrue(any("kv Test load" in line for line in logs.output))

        self.assertEqual(worker.m_store.get.call_count, 1)
        keys, addrs, sizes = worker.m_store.get.call_args.args
        self.assertEqual(len(keys), 1)
        self.assertEqual(addrs, [[target[1].data_ptr()]])
        self.assertEqual(sizes, [[target[1].numel() * target[1].element_size()]])
        self.assertIn("hash", keys[0])

    def test_start_load_kv_c128_aggregates_chunks_into_one_page(self):
        worker, target, metadata = self._make_c128_load_worker([0])
        chunks = [
            TransferChunkWithBlockId(
                raw_start=0,
                raw_end=512,
                value_start=0,
                value_end=4,
                target_block_index=0,
                key=PoolKey(KeyMetadata("hybrid-model", 0, 0, 0, 0), "hash0"),
                block_id=1,
            ),
            TransferChunkWithBlockId(
                raw_start=512,
                raw_end=1024,
                value_start=4,
                value_end=8,
                target_block_index=0,
                key=PoolKey(KeyMetadata("hybrid-model", 0, 0, 0, 0), "hash1"),
                block_id=1,
            ),
        ]
        worker.token_database.process_transfer_chunks_with_block_ids.return_value = chunks

        worker.start_load_kv(metadata)

        self.assertEqual(worker.m_store.get.call_count, 1)
        keys, addrs, sizes = worker.m_store.get.call_args.args
        self.assertEqual(len(keys), 1)
        self.assertIn("hash1", keys[0])
        self.assertEqual(addrs, [[target[1].data_ptr()]])
        self.assertEqual(sizes, [[target[1].numel() * target[1].element_size()]])
        self.assertTrue(torch.all(target == -1))

    def test_start_load_kv_c128_does_not_merge_failed_get(self):
        for get_result in ([1], None):
            with self.subTest(get_result=get_result):
                worker, target, metadata = self._make_c128_load_worker(get_result)

                worker.start_load_kv(metadata)

                self.assertTrue(torch.all(target == -1))

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

    def test_start_load_kv_prepares_layerwise_chunk_history(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
            KVPoolWorker,
        )

        worker = object.__new__(KVPoolWorker)
        worker.use_layerwise_range = True
        worker.current_layer = 0
        worker.layerwise_retrievers = []
        worker._range_load_layers_seen = set()
        worker._start_load_range_request = MagicMock()

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=8192,
            history_token_len=4096,
            block_ids=[0],
            block_hashes=["h0"],
            load_spec=None,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)

        worker.start_load_kv(meta)

        worker._start_load_range_request.assert_called_once_with(
            req,
            4096,
            require_all=True,
        )

    def test_start_range_save_rolls_back_every_request_and_can_retry(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_session import (
            ChunkStoreSession,
            StorePutRegistry,
            StoreReadLeaseRegistry,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
            KVPoolWorker,
            _LayerwiseRangeBlock,
        )

        worker = object.__new__(KVPoolWorker)
        worker.use_layerwise_range = True
        worker.group_uses_align_state = False
        worker.m_store = MagicMock()
        worker.m_store.exists.return_value = [0]
        worker.m_store.batch_put_session_start.return_value = [0]
        worker.m_store.batch_put_session_revoke.return_value = [0]
        worker.m_store.batch_get_session_end.return_value = 0
        worker._range_put_registry = StorePutRegistry(worker.m_store)
        worker._range_read_registry = StoreReadLeaseRegistry(worker.m_store)
        worker._range_execution_id = 0
        worker._range_save_started = False
        worker._range_sessions_by_request = {}
        worker._range_layer_futures = []
        worker._range_request_leases = {}
        worker._range_read_plans = {}
        worker._range_load_layers_seen = {"layer.0"}
        worker._range_request_generations = {}
        worker._next_range_request_generation = 0
        worker._range_object_sizes = {0: 128}
        worker._range_records = MagicMock(
            side_effect=lambda request, _token_len, group_id: [
                _LayerwiseRangeBlock(
                    key=f"key-{request.req_id}",
                    start=0,
                    end=128,
                    block_id=0,
                    group_id=group_id,
                )
            ]
        )
        worker._shard_range_save_records = MagicMock(
            side_effect=lambda _request, _group_id, records: records
        )

        leases = []

        def tracked_get_lease(request):
            lease = KVPoolWorker._get_range_lease(worker, request)
            if lease not in leases:
                leases.append(lease)
            return lease

        worker._get_range_lease = tracked_get_lease
        first = ReqMeta(
            req_id="first",
            token_len_chunk=128,
            block_ids=[0],
            can_save=True,
        )
        second = ReqMeta(
            req_id="second",
            token_len_chunk=128,
            block_ids=[1],
            can_save=True,
        )
        metadata = AscendConnectorMetadata(set(), set())
        metadata.add_request(first)
        metadata.add_request(second)
        original_start = ChunkStoreSession.start

        def fail_second(session, history_keys, claim):
            original_start(session, history_keys, claim)
            if session.request_chunk.request_id == "first":
                worker._range_read_plans["first"] = (
                    first,
                    session.request_lease,
                    [],
                )
            if session.request_chunk.request_id == "second":
                raise RuntimeError("second request initialization failed")

        with patch.object(ChunkStoreSession, "start", new=fail_second):
            with self.assertRaisesRegex(
                RuntimeError, "second request initialization failed"
            ):
                worker._start_range_save(metadata)

        revoked = {
            key
            for call in worker.m_store.batch_put_session_revoke.call_args_list
            for key in call.args[0]
        }
        self.assertEqual(revoked, {"key-first", "key-second"})
        self.assertTrue(all(lease.closed for lease in leases))
        self.assertEqual(worker._range_request_leases, {})
        self.assertEqual(worker._range_read_plans, {})
        self.assertEqual(worker._range_load_layers_seen, set())
        self.assertEqual(worker._range_request_generations, {})
        self.assertEqual(worker._range_put_registry._entries, {})
        self.assertEqual(worker._range_execution_id, 0)
        self.assertFalse(worker._range_save_started)
        self.assertEqual(worker._range_sessions_by_request, {})

        worker._get_range_lease = KVPoolWorker._get_range_lease.__get__(
            worker, KVPoolWorker
        )
        worker._start_range_save(metadata)
        self.assertEqual(worker._range_execution_id, 1)
        self.assertTrue(worker._range_save_started)
        self.assertEqual(
            set(worker._range_sessions_by_request), {"first", "second"}
        )

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

    def test_lookup_scheduler_layerwise_range_uses_cross_layer_keys(self):
        worker = self._make_worker()
        worker.use_layerwise_range = True
        worker._range_layout_fingerprint = "test-fingerprint"
        expected_keys = [
            worker._range_key(key, 0)
            for _, _, key in worker.token_database.process_tokens(
                32,
                ["h0", "h1"],
                kv_cache_group_id=0,
            )
        ]
        worker.m_store.exists.return_value = [1, 1]

        result = worker.lookup_scheduler(
            32,
            ["h0", "h1"],
            use_layerwise=True,
            use_layerwise_range=True,
        )

        self.assertEqual(result, 32)
        queried_keys = worker.m_store.exists.call_args.args[0]
        self.assertEqual(queried_keys, expected_keys)
        self.assertTrue(all("@layer_id:" not in key for key in queried_keys))

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


if __name__ == "__main__":
    unittest.main()
