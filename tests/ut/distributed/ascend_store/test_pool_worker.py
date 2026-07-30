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
    test.enterContext(patch(f"{module}.get_tensor_model_parallel_rank", return_value=tp_rank))
    test.enterContext(patch(f"{module}.get_tensor_model_parallel_world_size", return_value=tp_size))
    pcp_group = test.enterContext(patch(f"{module}.get_pcp_group"))
    pcp_group.return_value.world_size = 1
    test.enterContext(patch(f"{module}.get_decode_context_model_parallel_world_size", return_value=1))
    test.enterContext(patch(f"{module}.get_decode_context_model_parallel_rank", return_value=0))
    importlib = test.enterContext(patch(f"{module}.importlib"))
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

    def test_max_intersection_hit_position(self):
        cls = self._make_worker_class()
        cases = [
            ([[16, 32, 48]], 48),
            ([], 0),
            ([[16, 32, 48], [32, 48], [16, 32], [32, 48, 64]], 32),
        ]
        for hits, expected in cases:
            with self.subTest(hits=hits):
                self.assertEqual(cls._max_intersection_hit_position(hits), expected)

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
        worker.token_database.process_token_key_strings.side_effect = (
            lambda *args, chunk_filter, **kwargs: [(0, 128, "key", "ab" * 32)] if chunk_filter(0) else []
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

    def test_wait_for_save(self):
        for can_save in (True, False):
            with self.subTest(can_save=can_save):
                worker = self._make_worker()
                worker.kv_send_thread = MagicMock()
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=16,
                    block_ids=[0],
                    block_hashes=["h0"],
                    can_save=can_save,
                )
                meta = AscendConnectorMetadata(set())
                meta.add_request(req)
                worker.wait_for_save(meta)
                self.assertEqual(worker.kv_send_thread.add_request.called, can_save)

    def test_get_finished(self):
        for role, finished, expected in [("kv_producer", {"r1"}, {"r1"}), ("kv_consumer", set(), set())]:
            with self.subTest(role=role):
                worker = self._make_worker(kv_role=role)
                worker.kv_send_thread = MagicMock()
                worker.kv_send_thread.get_and_clear_finished_requests.return_value = finished
                done_s, done_r = worker.get_finished(finished, AscendConnectorMetadata(set()))
                self.assertEqual(done_s, expected)
                self.assertEqual(done_r, set())

    def test_lookup_scheduler(self):
        for exists, expected in [([1, 1], 32), ([1, 0], 16), (Exception("fail"), 0)]:
            for method_name in ("lookup", "lookup_scheduler"):
                with self.subTest(exists=exists, method=method_name):
                    worker = self._make_worker()
                    if isinstance(exists, Exception):
                        worker.m_store.exists.side_effect = exists
                    else:
                        worker.m_store.exists.return_value = exists
                    method = getattr(worker, method_name)
                    self.assertEqual(method(32, ["h0", "h1"], use_layerwise=False), expected)

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


class TestKVPoolWorkerGetBlockIdsWithLoadErrors(unittest.TestCase):
    """Test get_block_ids_with_load_errors method."""

    def _make_worker(self):
        return make_worker(self)

    def test_get_block_ids_with_load_errors(self):
        for invalid in ({1, 2, 3}, set()):
            with self.subTest(invalid=invalid):
                worker = self._make_worker()
                worker._invalid_block_ids = invalid.copy()
                self.assertEqual(worker.get_block_ids_with_load_errors(), invalid)
                self.assertEqual(worker._invalid_block_ids, set())


class TestKVPoolWorkerGetGroupTpSize(unittest.TestCase):
    """Test get_group_tp_size method."""

    def _make_worker(self):
        return make_worker(self, tp_size=4, num_kv_heads=8)

    def test_get_group_tp_size(self):
        for use_mla, align_state, expected in [(False, True, 4), (False, False, 4), (True, False, 1)]:
            with self.subTest(use_mla=use_mla, align_state=align_state):
                worker = self._make_worker()
                worker.use_mla = use_mla
                worker.group_uses_align_state = [align_state]
                self.assertEqual(worker.get_group_tp_size(0), expected)


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

    def test_process_save_for_layer_batch_skips(self):
        worker = self._make_worker()
        worker.process_layer_data([])
        for layer_tasks in worker.layer_save_tasks:
            self.assertEqual(layer_tasks, [])
        for layer_tasks in worker.layer_load_tasks:
            self.assertEqual(layer_tasks, [])

        cases = [
            {"can_save": False},
            {"can_save": True, "save_start_token": 16, "save_end_token": 16},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=32,
                    block_ids=[0, 1],
                    block_hashes=["h0", "h1"],
                    **overrides,
                )
                worker._process_save_for_layer_batch([req], 0)
                self.assertEqual(worker.layer_save_tasks[0], [])

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
