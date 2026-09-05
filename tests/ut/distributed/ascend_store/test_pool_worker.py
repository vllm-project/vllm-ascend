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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
    get_partial_block_index,
)


def start_patch(test: unittest.TestCase, *args, **kwargs):
    patcher = patch(*args, **kwargs)
    mocked = patcher.start()
    test.addCleanup(patcher.stop)
    return mocked


def make_worker(
    test: unittest.TestCase,
    *,
    kv_role="kv_producer",
    tp_rank=0,
    tp_size=1,
    num_kv_heads=1,
    num_layers=2,
    extra_config=None,
    use_layerwise=False,
    use_mla=False,
    enable_kv_events=False,
    num_hidden_layers=None,
):
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
    start_patch(test, f"{module}.get_tensor_model_parallel_rank", return_value=tp_rank)
    start_patch(test, f"{module}.get_tensor_model_parallel_world_size", return_value=tp_size)
    pcp_group = start_patch(test, f"{module}.get_pcp_group")
    pcp_group.return_value.world_size = 1
    start_patch(test, f"{module}.get_decode_context_model_parallel_world_size", return_value=1)
    start_patch(test, f"{module}.get_decode_context_model_parallel_rank", return_value=0)
    importlib = start_patch(test, f"{module}.importlib")
    importlib.import_module.return_value = MagicMock()

    config = MagicMock()
    config.model_config.model = "org/llama-7b"
    config.model_config.use_mla = use_mla
    config.model_config.hf_text_config = MagicMock(spec=[])
    if num_hidden_layers is not None:
        config.model_config.hf_text_config.num_hidden_layers = num_hidden_layers
    config.model_config.get_num_layers.return_value = num_layers
    config.model_config.get_total_num_kv_heads.return_value = num_kv_heads
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.rank = 0
    config.parallel_config.pipeline_parallel_size = 1
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_connector_extra_config = {
        "backend": "mooncake",
        **(extra_config or {}),
    }
    config.cache_config.block_size = 16
    config.kv_events_config = None
    if enable_kv_events:
        config.kv_events_config = MagicMock(enable_kv_cache_events=True)

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    return KVPoolWorker(config, use_layerwise=use_layerwise)


class TestKVPoolWorkerHelpers(unittest.TestCase):
    """Test the pure helper methods on KVPoolWorker without full init."""

    def _make_worker_class(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        return KVPoolWorker

    def test_check_all_layers_exists(self):
        cls = self._make_worker_class()
        cases = [
            ([1, 1, 1, 1, 1, 1], 3, [1, 1]),
            ([1, 1, 0, 1, 1, 1], 3, [0, 1]),
            ([0, 0, 0], 3, [0]),
        ]
        for exists, num_layers, expected in cases:
            with self.subTest(exists=exists):
                self.assertEqual(cls.check_all_layers_exists(None, exists, num_layers), expected)

    def test_uses_mamba_kv_cache_inside_uniform_group(self):
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

        cls = self._make_worker_class()
        mamba_spec = MambaSpec(block_size=384, shapes=((1,),), dtypes=(np.dtype("float32"),))
        uniform_spec = UniformTypeKVCacheSpecs.from_specs({"mamba.layer": mamba_spec})
        self.assertIsNotNone(uniform_spec)
        kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=uniform_spec)])

        self.assertTrue(cls._uses_mamba_kv_cache(True, kv_cache_config))

    def test_find_all_continuous_hit_positions(self):
        cls = self._make_worker_class()
        cases = [
            ([[1, 1, 0], [1, 0, 1]], [16, 32, 48], 3, [16]),
            ([[1, 1, 1], [1, 1, 1]], [16, 32, 48], 3, [16, 32, 48]),
            ([[0, 1], [1, 0]], [16, 32], 2, []),
            ([], [], 0, []),
        ]
        for exists, positions, count, expected in cases:
            with self.subTest(exists=exists):
                result = cls.find_all_continuous_hit_positions(exists, positions, count, 48, 16)
                self.assertEqual(result, expected)

    def test_find_all_discontinuous_hit_positions(self):
        cls = self._make_worker_class()
        positions = [16, 32, 48, 64, 80, 96]
        cases = [
            ([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]], 128, [48, 96]),
            ([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0]], 128, [48]),
            ([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]], 64, [48]),
        ]
        for exists, token_len, expected in cases:
            with self.subTest(exists=exists, token_len=token_len):
                result = cls.find_all_discontinuous_hit_positions(exists, positions, 6, token_len, 16)
                self.assertEqual(result, expected)

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

    def test_wait_for_layer_load_fallback_waits_for_reuse(self):
        cls = self._make_worker_class()
        worker = cls.__new__(cls)
        worker.current_layer = 0
        worker.num_layers = 1
        worker.layer_load_tasks = [[]]
        worker.prefetch_layer_map = {}
        worker.layer_load_finished_events = [threading.Event()]
        worker.kv_recv_thread = MagicMock()
        worker.external_slot_release_waiter = MagicMock()
        worker._submit_ready_layer_loads = MagicMock()

        worker.wait_for_layer_load()

        worker.external_slot_release_waiter.assert_called_once_with(0)

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

    def test_partial_prefill_block_index_boundaries(self):
        self.assertEqual(get_partial_block_index(20, 16, 1, True), 1)
        self.assertEqual(get_partial_block_index(32, 16, 1, True), 1)
        self.assertIsNone(get_partial_block_index(32, 16, 2, True))
        self.assertIsNone(get_partial_block_index(20, 16, 1, False))

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

    def test_external_coordinator_lookup_uses_only_lookup_mask(self):
        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.hash_block_size = 128
        worker.num_kv_cache_groups = 1
        worker.cache_coordinator = MagicMock()
        worker.cache_coordinator.lcm_block_size = 128
        worker.cache_coordinator.lookup_mask.return_value = ([True],)
        worker.cache_coordinator.store_mask.return_value = ([False],)
        worker.cache_coordinator.find_longest_cache_hit.return_value = ((), 128)
        worker.m_store = MagicMock()
        worker.m_store.exists.return_value = [1]

        worker.token_database = MagicMock()
        worker.token_database.get_block_size.return_value = 128
        worker.token_database.group_cache_families = {"kv": {0: "default"}}
        worker.token_database.process_token_key_strings.side_effect = lambda *args, chunk_filter, **kwargs: (
            [(0, 128, "key", "ab" * 32)] if chunk_filter(0) else []
        )

        hit = worker._lookup_with_coordinator(
            128,
            [b"h0"],
            [0],
            use_layerwise=False,
            include_all_ranks=False,
        )

        self.assertEqual(hit, 128)
        worker.cache_coordinator.lookup_mask.assert_called_once_with(128)
        worker.cache_coordinator.store_mask.assert_not_called()
        worker.m_store.exists.assert_called_once_with(["key"])
        worker.cache_coordinator.find_longest_cache_hit.assert_called_once()
        self.assertFalse(worker.cache_coordinator.find_longest_cache_hit.call_args.kwargs["apply_eagle"])
        worker.token_database.process_tokens.assert_not_called()

    def test_layerwise_multi_group_layout_includes_mtp(self):
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        cls = self._make_worker_class()
        worker = object.__new__(cls)
        worker.num_layers = 4
        worker.num_kv_cache_groups = 2
        worker.hf_config = SimpleNamespace(num_hidden_layers=4)
        worker.use_layerwise_transfer = True
        worker._extra_config = {"layerwise_num_shared_buffers": 2}
        main_spec = FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
        )
        indexer_spec = FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float16,
        )
        worker.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[
                        *(f"model.layers.{layer}.self_attn.attn" for layer in range(4)),
                        "model.mtp.0.self_attn.attn",
                    ],
                    kv_cache_spec=main_spec,
                ),
                SimpleNamespace(
                    layer_names=[
                        *(f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in range(4)),
                    ],
                    kv_cache_spec=indexer_spec,
                ),
            ]
        )

        worker._init_layerwise_config()

        self.assertEqual(worker.num_layers, 5)
        self.assertEqual(worker.physical_layer_to_group_layers[4], [(0, 4)])
        self.assertTrue(worker.layerwise_offload)
        self.assertEqual(worker.independent_layers, [0])
        self.assertEqual(len(worker.layer_load_tasks), 5)
        self.assertEqual(len(worker.layer_save_tasks), 5)


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

    def _make_worker(self, kv_role="kv_producer", extra_config=None, use_layerwise=False):
        self._patch_all()
        config = self._make_config(kv_role=kv_role, extra_config=extra_config)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(config, use_layerwise=use_layerwise)
        worker._uses_vllm_mooncake_connector = MagicMock(return_value=False)
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

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.threading.Event")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreRecvingThread")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreSendingThread")
    def test_transfer_threads_use_grouped_block_sizes(self, send_thread, recv_thread, event):
        worker = self._make_worker(kv_role="kv_both", extra_config={"backend": "mooncake", "load_async": True})
        worker.grouped_block_size = [128, 128, 128, 128, 8, 32]

        worker._start_kv_transfer_threads()

        self.assertEqual(send_thread.call_args.args[2], worker.grouped_block_size)
        self.assertEqual(recv_thread.call_args.args[2], worker.grouped_block_size)
        event.return_value.wait.assert_called()

    def test_register_kv_caches_initializes_layerwise_memcache(self):
        worker = self._make_worker(extra_config={"backend": "memcache"}, use_layerwise=True)
        fake_cache = MagicMock()
        fake_cache.shape = [100, 16, 8, 64]
        fake_cache.element_size.return_value = 2
        fake_cache.data_ptr.return_value = 10000
        worker._transfer_threads_started = True

        worker.register_kv_caches({"layer.0": (fake_cache, fake_cache)})

        worker.m_store.ensure_initialized.assert_called_once_with()
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

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreRecvingThread.start",
        autospec=True,
    )
    def test_async_load_failure_is_reported_by_worker(self, start_thread):
        worker = self._make_worker(kv_role="kv_consumer", extra_config={"load_async": True})
        worker.m_store.get = MagicMock()
        worker.token_database.set_group_buffers({0: [1000]}, {0: [160]})
        worker.m_store.get.return_value = [1]
        start_thread.side_effect = lambda thread: thread.ready_event.set()
        worker._start_kv_transfer_threads()

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[7],
            block_hashes=["h0"],
            load_spec=LoadSpec(0, 16, can_load=True, token_len=16),
        )
        meta = AscendConnectorMetadata(set())
        meta.add_request(req)
        worker.start_load_kv(meta)

        recv_thread = worker.kv_recv_thread
        recv_thread._handle_request(recv_thread.request_queue.get_nowait())
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7})
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_start_load_kv(self):
        cases = [
            (16, [0], ["h0"], LoadSpec(0, 16, True, token_len=16), True),
            (64, [99], ["h0", "h1", "h2", "h3"], LoadSpec(0, 64, True, token_len=64), True),
            (16, [0], ["h0"], None, False),
        ]
        for token_len, block_ids, hashes, load_spec, should_load in cases:
            with self.subTest(token_len=token_len, block_ids=block_ids, load_spec=load_spec):
                worker = self._make_worker()
                worker.m_store.get = MagicMock()
                worker.token_database.set_group_buffers({0: [1000]}, {0: [160]})
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=token_len,
                    block_ids=block_ids,
                    block_hashes=hashes,
                    load_spec=load_spec,
                )
                meta = AscendConnectorMetadata(set())
                meta.add_request(req)
                worker.start_load_kv(meta)
                self.assertEqual(worker.m_store.get.called, should_load)
                if block_ids == [99]:
                    _, addrs, sizes = worker.m_store.get.call_args.args
                    self.assertEqual(addrs, [[1000 + 99 * 160]])
                    self.assertEqual(sizes, [[160]])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVCacheStoreRecvingThread")
    def test_async_recv_thread_shares_invalid_block_state(self, recv_thread_cls):
        worker = self._make_worker(
            kv_role="kv_consumer",
            extra_config={"backend": "mooncake", "load_async": True},
        )
        recv_thread = MagicMock()

        def create_recv_thread(*args, **kwargs):
            args[6].set()
            return recv_thread

        recv_thread_cls.side_effect = create_recv_thread

        worker._start_kv_transfer_threads()

        kwargs = recv_thread_cls.call_args.kwargs
        self.assertIs(kwargs["invalid_block_ids"], worker._invalid_block_ids)
        self.assertIs(
            kwargs["invalid_block_ids_lock"],
            worker._invalid_block_ids_lock,
        )
        kwargs["invalid_block_ids"].add(7)
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7})

    def test_wait_for_save_waits_for_save(self):
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
        worker.kv_send_thread.request_queue.join.assert_called_once()

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
        worker.kv_send_thread.request_queue.join.assert_not_called()

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

    def test_lookup_all_cached(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 1]
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 32)

    def test_lookup_partial(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 0]
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)

    def test_lookup_exception(self):
        worker = self._make_worker()
        worker.m_store.exists.side_effect = Exception("fail")
        result = worker.lookup(32, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    def test_lookup_layerwise(self):
        worker = self._make_worker()
        worker.m_store.exists.return_value = [1, 1, 1, 1]
        for method in (worker.lookup, worker.lookup_scheduler):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(32, ["h0", "h1"], use_layerwise=True), 32)

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


class TestKVPoolWorkerBuildConnectorWorkerMeta(unittest.TestCase):
    """Test build_connector_worker_meta method."""

    def _make_worker(self):
        return make_worker(self)

    def test_build_connector_worker_meta(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread

        cases = [(False, None, None), (True, None, None), (True, {}, None), (True, {1: 2}, {1: 2})]
        for use_mamba, events, expected in cases:
            with self.subTest(use_mamba=use_mamba, events=events):
                worker = self._make_worker()
                worker.use_mamba = use_mamba
                if events is not None:
                    worker.kv_send_thread = MagicMock(spec=KVCacheStoreSendingThread)
                    worker.kv_send_thread.get_completed_events.return_value = events
                else:
                    worker.kv_send_thread = None
                result = worker.build_connector_worker_meta()
                self.assertEqual(None if result is None else result.completed_events, expected)


class TestKVPoolWorkerGetFinishedAsync(unittest.TestCase):
    """Test get_finished with async recv thread."""

    def _make_worker(self, kv_role="kv_consumer"):
        return make_worker(self, kv_role=kv_role, extra_config={"load_async": True})

    def test_get_finished_async_recv_thread(self):
        worker = self._make_worker(kv_role="kv_consumer")
        worker.load_async = True

        recv_thread = MagicMock()
        recv_thread.get_and_clear_finished_requests.return_value = {"r1"}
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = None

        loading_req_ids = {"r1"}
        meta = AscendConnectorMetadata(set(), loading_req_ids=loading_req_ids)
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, {"r1"})
        recv_thread.get_and_clear_finished_requests.assert_called_once_with(loading_req_ids)

        recv_thread.reset_mock()
        recv_thread.get_and_clear_finished_requests.return_value = set()
        meta = AscendConnectorMetadata({"r_preempted"}, loading_req_ids=set())
        worker.get_finished(set(), meta)
        recv_thread.discard_finished_requests.assert_called_once_with({"r_preempted"})

    def test_get_finished_layerwise_send_thread(self):
        worker = self._make_worker(kv_role="kv_producer")
        worker.use_layerwise = True

        send_thread = MagicMock()
        send_thread.get_and_clear_finished_requests.return_value = set()
        worker.kv_send_thread = send_thread
        worker.kv_recv_thread = None

        meta = AscendConnectorMetadata(set())
        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, set())
        send_thread.get_and_clear_finished_requests.assert_called_once_with()


class TestKVPoolWorkerStartLoadKVAsync(unittest.TestCase):
    """Test start_load_kv with load_async=True."""

    def _make_worker(self):
        worker = make_worker(self, kv_role="kv_consumer", extra_config={"load_async": True})
        worker.load_async = True
        return worker

    def test_start_load_kv_async(self):
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
        meta = AscendConnectorMetadata(set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        recv_thread.add_request.assert_called_once_with(req)

        recv_thread.reset_mock()
        worker = self._make_worker()
        worker.kv_recv_thread = recv_thread
        worker.start_load_kv(AscendConnectorMetadata(set()))
        recv_thread.add_request.assert_not_called()


class TestKVPoolWorkerProcessLayerData(unittest.TestCase):
    """Test process_layer_data and related layerwise methods."""

    def _make_worker(self):
        return make_worker(self)

    def _make_gva_worker(self, num_groups=1):
        worker = make_worker(self, extra_config={"backend": "memcache"}, use_layerwise=True)
        worker.layerwise_offload = True
        worker.num_kv_cache_groups = num_groups
        worker.grouped_block_size = [16] * num_groups
        worker.kv_cache_group_families = ["default"] * num_groups
        worker.group_block_len = {group_id: [64] for group_id in range(num_groups)}
        worker.group_num_layers = {group_id: 1 for group_id in range(num_groups)}
        worker.hash_block_size = 16
        worker.page_size_bytes = 64
        worker.head_or_tp_rank = 0
        worker.m_store = MagicMock()
        return worker

    @staticmethod
    def _make_gva_request(num_groups=1, load_spec=None, can_save=None):
        block_ids_by_group = [[7 + group_id] for group_id in range(num_groups)]
        return ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            save_start_token=0,
            save_end_token=16,
            target_token_len=16,
            block_ids=block_ids_by_group[0],
            block_ids_by_group=block_ids_by_group,
            block_hashes=["h0"],
            can_save=can_save,
            load_spec=load_spec,
            block_ids_np=np.asarray(block_ids_by_group[0], dtype=np.int64),
            block_ids_by_group_np=[np.asarray(block_ids, dtype=np.int64) for block_ids in block_ids_by_group],
        )

    def test_set_external_slot_release_waiter_gated_on_layerwise_transfer(self):
        waiter = MagicMock()

        worker = self._make_worker()
        worker.use_layerwise_transfer = False
        self.assertFalse(worker.set_external_slot_release_waiter(waiter))
        self.assertIsNone(worker.external_slot_release_waiter)

        worker.use_layerwise_transfer = True
        self.assertTrue(worker.set_external_slot_release_waiter(waiter))
        self.assertIs(worker.external_slot_release_waiter, waiter)

    def test_set_external_slot_release_waiter_updates_running_recv_thread(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreLayerRecvingThread,
        )

        waiter = MagicMock()

        worker = self._make_worker()
        worker.use_layerwise_transfer = True
        worker.kv_recv_thread = MagicMock(spec=KVCacheStoreLayerRecvingThread)
        self.assertTrue(worker.set_external_slot_release_waiter(waiter))
        # A waiter registered after the receive thread started is handed
        # over to the thread directly, not just stored on the worker.
        self.assertIs(worker.kv_recv_thread.external_slot_release_waiter, waiter)

    def test_process_layer_data_empty_requests(self):
        worker = self._make_worker()
        worker.process_layer_data([])
        for layer_tasks in worker.layer_save_tasks:
            self.assertEqual(layer_tasks, [])
        for layer_tasks in worker.layer_load_tasks:
            self.assertEqual(layer_tasks, [])

    def test_empty_layerwise_step_reowns_task_lists(self):
        worker = self._make_worker()
        worker.use_layerwise = True
        old_save_tasks = worker.layer_save_tasks
        old_load_tasks = worker.layer_load_tasks

        worker.start_load_kv(AscendConnectorMetadata(set(), set()))

        for layer_id in range(worker.num_layers):
            self.assertIsNot(worker.layer_save_tasks[layer_id], old_save_tasks[layer_id])
            self.assertIsNot(worker.layer_load_tasks[layer_id], old_load_tasks[layer_id])

    def test_layerwise_load_is_prepared_before_next_save_allocation(self):
        worker = self._make_worker()
        worker.num_layers = 0
        call_order = []
        worker._prepare_load_gvas = MagicMock(side_effect=lambda requests: call_order.append("load"))
        worker._alloc_gvas_for_save = MagicMock(side_effect=lambda requests: call_order.append("save"))
        worker._build_shared_save_data = MagicMock()
        worker._build_shared_load_data = MagicMock()

        worker.process_layer_data([MagicMock()])

        self.assertEqual(call_order, ["load", "save"])

    def test_process_layer_data_reowns_task_lists_before_populating(self):
        worker = self._make_worker()
        old_save_tasks = worker.layer_save_tasks
        old_load_tasks = worker.layer_load_tasks
        save_marker = MagicMock()
        load_marker = MagicMock()
        worker._process_save_for_layer_batch = MagicMock(
            side_effect=lambda _requests, layer_id, *_args: worker.layer_save_tasks[layer_id].append(save_marker)
        )
        worker._process_load_for_layer_batch = MagicMock(
            side_effect=lambda _requests, layer_id, *_args: worker.layer_load_tasks[layer_id].append(load_marker)
        )
        worker._prepare_load_gvas = MagicMock()
        worker._alloc_gvas_for_save = MagicMock()
        worker._build_shared_save_data = MagicMock()
        worker._build_shared_load_data = MagicMock()

        worker.process_layer_data([MagicMock()])

        for layer_id in range(worker.num_layers):
            self.assertIsNot(worker.layer_save_tasks[layer_id], old_save_tasks[layer_id])
            self.assertIsNot(worker.layer_load_tasks[layer_id], old_load_tasks[layer_id])
            old_save_tasks[layer_id].clear()
            old_load_tasks[layer_id].clear()
            self.assertEqual(worker.layer_save_tasks[layer_id], [save_marker])
            self.assertEqual(worker.layer_load_tasks[layer_id], [load_marker])

    def test_build_shared_save_data_marks_last_actual_task(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreLayerSendingThread,
        )

        worker = self._make_worker()
        worker.num_layers = 3
        worker.num_kv_cache_groups = 1
        first_task = LayerTransferTask(layer_id=0, block_ranges=[])
        last_task = LayerTransferTask(layer_id=1, block_ranges=[])
        worker.layer_save_tasks = [[first_task], [last_task], []]
        shared = SharedBlockData(
            block_ids_arr=np.asarray([0]),
            block_gvas_arr=np.asarray([100]),
            req_ids=["r1"],
            is_last_chunks=[True],
            save_keys=["k0"],
        )
        send_thread = object.__new__(KVCacheStoreLayerSendingThread)
        send_thread.build_shared_data = MagicMock(return_value=shared)
        worker.kv_send_thread = send_thread

        worker._build_shared_save_data()

        self.assertEqual(first_task.write_finish_keys, [])
        self.assertEqual(last_task.write_finish_keys, ["k0"])

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

    def test_process_load_for_layer_batch_skips(self):
        for load_spec in (None, LoadSpec(0, 0, can_load=False, token_len=0)):
            with self.subTest(load_spec=load_spec):
                worker = self._make_worker()
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=32,
                    block_ids=[0, 1],
                    block_hashes=["h0", "h1"],
                    load_spec=load_spec,
                )
                worker._process_load_for_layer_batch([req], 0)
                self.assertEqual(worker.layer_load_tasks[0], [])

    def test_reused_layer_loads_full_cached_prefix(self):
        worker = self._make_worker()
        worker.layerwise_offload = True
        worker.independent_layers = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=32,
                can_load=True,
                token_len=32,
            ),
        )

        worker._process_load_for_layer_batch([request], 0)
        worker._process_load_for_layer_batch([request], 1)

        independent_range = worker.layer_load_tasks[0][0].block_ranges[0]
        reused_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual((independent_range.start_block, independent_range.end_block), (1, 2))
        self.assertEqual((reused_range.start_block, reused_range.end_block), (0, 2))

    def test_mtp_load_uses_safe_extent_not_store_skip_extent(self):
        worker = self._make_worker()
        worker.use_eagle = True
        worker.layerwise_offload = True
        worker.independent_layers = [0, 1]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
                kvpool_store_skip_tokens=32,
            ),
        )

        worker._process_load_for_layer_batch([request], 1)

        self.assertEqual(worker.layer_load_tasks[1], [])

    def test_mtp_gva_prepare_uses_safe_extent_not_store_skip_extent(self):
        worker = self._make_gva_worker()
        worker.use_eagle = True
        key_info = MagicMock()
        key_info.size.return_value = 64
        key_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.return_value = [key_info]
        worker.m_store.batch_add_lease.return_value = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1]],
            block_ids_by_group_np=[np.asarray([0, 1], dtype=np.int64)],
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
                kvpool_store_skip_tokens=32,
            ),
        )

        worker._prepare_load_gvas([request])

        queried_keys = worker.m_store.batch_get_key_info.call_args.args[0]
        self.assertEqual(len(queried_keys), 1)

    def test_full_pool_hit_uses_verified_extent(self):
        worker = self._make_gva_worker()
        worker.independent_layers = [0]
        key_info = MagicMock()
        key_info.size.return_value = 64
        key_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.return_value = [key_info]
        worker.m_store.batch_add_lease.return_value = [0]
        request = self._make_gva_request(
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=15,
                can_load=True,
                kvpool_store_skip_tokens=16,
            ),
            can_save=True,
        )

        worker._prepare_load_gvas([request])
        worker._alloc_gvas_for_save([request])
        worker._process_load_for_layer_batch([request], 1)
        worker._process_save_for_layer_batch([request], 1)

        queried_keys = worker.m_store.batch_get_key_info.call_args.args[0]
        self.assertEqual(len(queried_keys), 1)
        self.assertNotIn("@partial@", queried_keys[0])
        worker.m_store.batch_alloc.assert_not_called()
        load_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual((load_range.start_block, load_range.end_block), (0, 1))
        self.assertIsNone(load_range.partial_block_index)
        self.assertEqual(worker.layer_save_tasks[1], [])

    def test_partial_prefill_is_saved_and_loaded_for_reused_layer(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
            pool_worker as _pool_worker,
        )

        self.assertIsNotNone(_pool_worker)
        worker = make_worker(self, extra_config={"backend": "memcache"}, use_layerwise=True)
        worker.layerwise_offload = True
        worker.independent_layers = [0]
        worker.num_kv_cache_groups = 1
        worker.grouped_block_size = [16]
        worker.kv_cache_group_families = ["default"]
        worker.group_block_len = {0: [64]}
        worker.group_num_layers = {0: 1}
        worker.hash_block_size = 16
        worker.page_size_bytes = 64
        worker.head_or_tp_rank = 0
        worker._allocated_gvas = {}
        worker.m_store = MagicMock()
        worker.m_store.batch_alloc.return_value = [101]

        save_request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            save_start_token=16,
            save_end_token=16,
            target_token_len=20,
            num_prompt_tokens=32,
            block_ids=[0, 1],
            block_hashes=["h0"],
            can_save=True,
            block_ids_np=np.asarray([0, 1], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([0, 1], dtype=np.int64)],
        )
        worker._alloc_gvas_for_save([save_request])
        worker._process_save_for_layer_batch([save_request], 1)

        self.assertIsNotNone(save_request.save_keys)
        assert save_request.save_keys is not None
        partial_key = save_request.save_keys[0]
        self.assertIn("@partial@r1@0@1@20@", partial_key)
        self.assertEqual(save_request.partial_save_gva_per_group, [101])
        save_range = worker.layer_save_tasks[1][0].block_ranges[0]
        self.assertEqual(save_range.partial_block_index, 1)

        normal_info = MagicMock()
        normal_info.size.return_value = 64
        normal_info.gva_list.return_value = [201]
        partial_info = MagicMock()
        partial_info.size.return_value = 64
        partial_info.gva_list.return_value = [202]
        worker.m_store.batch_get_key_info.return_value = [
            normal_info,
            partial_info,
        ]
        worker.m_store.batch_add_lease.return_value = [0, 0]

        load_request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            target_token_len=24,
            num_prompt_tokens=32,
            block_ids=[0, 1],
            block_hashes=["h0"],
            load_spec=LoadSpec(
                vllm_cached_tokens=20,
                kvpool_cached_tokens=20,
                can_load=True,
            ),
            block_ids_np=np.asarray([0, 1], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([0, 1], dtype=np.int64)],
        )
        worker._prepare_load_gvas([load_request])
        worker._process_load_for_layer_batch([load_request], 0)
        worker._process_load_for_layer_batch([load_request], 1)

        queried_keys = worker.m_store.batch_get_key_info.call_args.args[0]
        self.assertIn(partial_key, queried_keys)
        self.assertNotIn(partial_key, worker._allocated_gvas)
        self.assertEqual(load_request.partial_load_gva_per_group, [202])
        self.assertEqual(worker.layer_load_tasks[0], [])
        block_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual(
            (
                block_range.start_block,
                block_range.end_block,
                block_range.partial_block_index,
            ),
            (0, 1, 1),
        )

    def test_layerwise_lease_failure_is_not_copied(self):
        worker = self._make_gva_worker()
        key_info = MagicMock()
        key_info.size.return_value = 64
        key_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.return_value = [key_info]
        worker.m_store.batch_add_lease.return_value = [-1]
        request = self._make_gva_request(
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
            ),
        )

        worker._prepare_load_gvas([request])

        self.assertEqual(request.load_block_gvas_by_group_np[0].tolist(), [0])
        self.assertEqual(request.load_keys, [])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7})

    def test_partial_lease_retries_until_snapshot_is_readable(self):
        worker = self._make_gva_worker()
        full_info = MagicMock()
        full_info.size.return_value = 64
        full_info.gva_list.return_value = [201]
        partial_info = MagicMock()
        partial_info.size.return_value = 64
        partial_info.gva_list.return_value = [202]
        worker.m_store.batch_get_key_info.return_value = [
            full_info,
            partial_info,
        ]
        worker.m_store.batch_add_lease.side_effect = [
            [0, -3101],
            [0],
        ]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            target_token_len=24,
            block_ids=[7, 8],
            block_hashes=["h0"],
            load_spec=LoadSpec(
                vllm_cached_tokens=20,
                kvpool_cached_tokens=20,
                can_load=True,
            ),
            block_ids_np=np.asarray([7, 8], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([7, 8], dtype=np.int64)],
        )

        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.sleep") as sleep:
            worker._prepare_load_gvas([request])

        partial_key = worker._make_layerwise_partial_key(request, 0, 1, 20)
        self.assertEqual(
            worker.m_store.batch_add_lease.call_args_list[1].args[0],
            [partial_key],
        )
        sleep.assert_called_once()
        self.assertEqual(request.load_keys, [worker._make_layerwise_full_key(0, "h0"), partial_key])
        self.assertEqual(request.partial_load_gva_per_group, [202])
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_multi_group_load_failure_stops_before_forward(self):
        worker = self._make_gva_worker(2)
        valid_info = MagicMock()
        valid_info.size.return_value = 64
        valid_info.gva_list.return_value = [201]
        missing_info = MagicMock()
        missing_info.size.return_value = 0
        missing_info.gva_list.return_value = []
        worker.m_store.batch_get_key_info.side_effect = [
            [valid_info],
            [missing_info],
        ]
        worker.m_store.batch_add_lease.return_value = [0]
        request = self._make_gva_request(
            num_groups=2,
            load_spec=LoadSpec(
                vllm_cached_tokens=16,
                kvpool_cached_tokens=16,
                can_load=True,
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "multi-group KV load failed",
        ):
            worker._prepare_load_gvas([request])

        group0_key = worker._make_layerwise_full_key(0, "h0")
        worker.m_store.batch_remove_lease.assert_called_once_with([group0_key])

    def test_worker_physical_layer_index_supports_mtp_layers_namespace(self):
        worker = self._make_worker()

        self.assertEqual(
            worker._extract_physical_layer_index(
                "mtp.layers.0.self_attn",
            ),
            worker.num_layers,
        )

    def test_evicted_allocated_gva_is_reallocated(self):
        worker = self._make_gva_worker()
        key = worker._make_layerwise_full_key(0, "h0")
        worker._allocated_gvas[key] = 101
        worker.m_store.batch_is_exist.return_value = [0]
        worker.m_store.batch_alloc.return_value = [202]
        request = self._make_gva_request(can_save=True)

        worker._alloc_gvas_for_save([request])

        worker.m_store.batch_alloc.assert_called_once_with([key], [64])
        self.assertEqual(worker._allocated_gvas[key], 202)
        self.assertEqual(request.block_gvas_by_group_np[0].tolist(), [202])

    def test_partial_decode_is_saved_and_loaded_for_reused_layer(self):
        worker = self._make_worker()
        worker.layerwise_offload = True
        worker.independent_layers = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            save_start_token=32,
            save_end_token=32,
            target_token_len=34,
            num_prompt_tokens=32,
            block_ids=[0, 1, 2],
            block_hashes=["h0", "h1"],
            can_save=True,
            load_spec=LoadSpec(
                vllm_cached_tokens=33,
                kvpool_cached_tokens=33,
                can_load=True,
            ),
            partial_save_gva_per_group=[301],
            partial_load_gva_per_group=[302],
        )

        worker._process_save_for_layer_batch([request], 1)
        worker._process_load_for_layer_batch([request], 1)

        save_range = worker.layer_save_tasks[1][0].block_ranges[0]
        load_range = worker.layer_load_tasks[1][0].block_ranges[0]
        self.assertEqual(save_range.partial_block_index, 2)
        self.assertEqual(load_range.partial_block_index, 2)


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
        use_mla=False,
    ):
        patches = self._patches(tp_rank=tp_rank, tp_size=tp_size)
        self._start(patches)
        try:
            cfg = self._make_vllm_config(
                kv_role=kv_role, extra_config=extra_config, num_kv_heads=num_kv_heads, use_sparse=use_sparse
            )
            cfg.model_config.use_mla = use_mla
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            return KVPoolWorker(cfg, use_layerwise=use_layerwise)
        finally:
            for p in patches:
                p.stop()

    def _make_strided_worker(self, tp_rank=0):
        worker = self._make_worker(
            tp_rank=tp_rank,
            extra_config={"backend": "mooncake", "prefill_tp_size": 4},
        )
        worker.block_size = 4
        worker.group_kv_caches_base_addr = {0: [0]}
        worker.group_block_len = {0: [16]}
        worker.group_block_stride = {0: [16]}
        worker.sub_size_bytes = 2
        worker.token_database.block_size = [4]
        worker.token_database.hash_block_size = 4
        return worker

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
        worker._uses_vllm_mooncake_connector = MagicMock(return_value=False)
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

    def test_tp_mismatch_disabled(self):
        cases = [
            ({"backend": "mooncake"}, False),
            ({"backend": "mooncake", "prefill_tp_size": 2}, False),
            ({"backend": "mooncake", "prefill_tp_size": 4}, True),
        ]
        for extra_config, use_mla in cases:
            with self.subTest(extra_config=extra_config, use_mla=use_mla):
                worker = self._make_worker(extra_config=extra_config, use_mla=use_mla)
                self.assertFalse(worker.tp_mismatch)
                self.assertEqual(worker.num_sub_keys, 1)

    def test_tp_mismatch_rejects_incompatible_layouts(self):
        for options in ({"use_sparse": True}, {"use_layerwise": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._make_worker(
                    extra_config={"backend": "mooncake", "prefill_tp_size": 4},
                    **options,
                )

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

    def test_build_tp_mismatch_keys_and_addrs(self):
        worker = self._make_strided_worker(tp_rank=1)

        keys, addrs, sizes, block_ids = worker._build_tp_mismatch_keys_and_addrs(
            block_hashes=[b"h0", b"h1"], block_ids=[10, 11], token_len=8, mask_num=0
        )
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(addrs), 4)
        self.assertEqual(len(sizes), 4)
        self.assertEqual(len(block_ids), 4)
        self.assertIn("@head_or_tp_rank:2", keys[0])
        self.assertIn("@head_or_tp_rank:3", keys[1])

        keys, addrs, sizes, block_ids = worker._build_tp_mismatch_keys_and_addrs(
            block_hashes=[b"h0", b"h1"], block_ids=[10], token_len=8, mask_num=0
        )
        self.assertEqual(len(keys), 2)
        self.assertEqual(len(addrs), 2)
        self.assertEqual(len(sizes), 2)
        self.assertEqual(block_ids, [10, 10])
        self.assertTrue(keys[0].endswith(f"@{b'h1'.hex()}"))

    def test_load_kv_tp_mismatch_calls_backend_get(self):
        worker = self._make_strided_worker()
        worker.m_store = MagicMock()
        worker.m_store.get.return_value = [0]  # success

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

    def test_store_kv_tp_mismatch_decrements_on_success_and_error(self):
        for put_error in (None, RuntimeError("put failed")):
            with self.subTest(put_error=put_error):
                worker = self._make_strided_worker()
                worker.m_store = MagicMock()
                worker.m_store.put.side_effect = put_error
                worker.enable_kv_events = False
                send_thread = MagicMock()
                send_thread.is_stored_request.return_value = True
                send_thread.lookup.return_value = [False, True]
                worker.kv_send_thread = send_thread
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=4,
                    block_ids_by_group=[[5]],
                    block_hashes=[b"h0"],
                    current_event=None,
                )

                if put_error:
                    with self.assertRaises(RuntimeError):
                        worker._store_kv_tp_mismatch(req)
                else:
                    worker._store_kv_tp_mismatch(req)
                    self.assertEqual(len(worker.m_store.put.call_args.args[0]), 1)
                send_thread.dec_stored_request.assert_called_once_with("r1")


if __name__ == "__main__":
    unittest.main()
