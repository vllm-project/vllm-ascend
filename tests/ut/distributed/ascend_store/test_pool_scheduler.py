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

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LoadSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
    LookupKeyClient,
    get_zmq_rpc_path_lookup,
)


@pytest.fixture(autouse=True)
def _patch_pool_scheduler_importlib():
    """KVPoolScheduler resolves its backend dynamically via
    ``importlib.import_module``; point it at a MagicMock so the scheduler's
    ``store_scheduler`` is a mock (the heavy real backends are exercised
    separately in test_backend.py). Scoped to this module so test_backend.py,
    which imports the real backend classes and uses ``mock.patch`` (itself
    backed by importlib.import_module), is unaffected.
    """
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as mock_importlib:
        mock_importlib.import_module.return_value = MagicMock()
        yield


class TestGetZmqRpcPathLookup(unittest.TestCase):
    def test_default_port(self):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}
        result = get_zmq_rpc_path_lookup(config)
        self.assertIn("lookup_rpc_port_0", result)
        self.assertIn("dp_rank0", result)

    def test_lookup_rpc_port(self):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 1
        config.kv_transfer_config.kv_connector_extra_config = {"lookup_rpc_port": 5555}
        result = get_zmq_rpc_path_lookup(config)
        self.assertIn("lookup_rpc_port_5555", result)
        self.assertIn("dp_rank1", result)

    def test_mooncake_rpc_port_fallback(self):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {"mooncake_rpc_port": 6666}
        result = get_zmq_rpc_path_lookup(config)
        self.assertIn("lookup_rpc_port_6666", result)


class TestKVPoolScheduler(unittest.TestCase):
    def _make_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = block_size
        config.cache_config.hash_block_size = block_size
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        return config

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_consumer_no_load(self, mock_client_cls):
        config = self._make_config(kv_role="kv_consumer")
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        result = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(result, (0, False))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_too_short(self, mock_client_cls):
        config = self._make_config(block_size=64)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.prompt_token_ids = list(range(32))
        result = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(result, (0, False))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_hit(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, is_async = scheduler.get_num_new_matched_tokens(request, 16)
        self.assertEqual(need, 32)  # 48 - 16
        self.assertFalse(is_async)
        self.assertIn("r1", scheduler.load_specs)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_all_hit(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        # When external hit equals num_tokens, reduce by 1
        mock_client_cls.return_value.lookup.return_value = 64

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, _ = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(need, 63)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_less_than_computed(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 16

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, _ = scheduler.get_num_new_matched_tokens(request, 32)
        self.assertEqual(need, 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_update_state_after_alloc_no_load_spec(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertIn("r1", scheduler._unfinished_request_ids)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_update_state_after_alloc_with_load(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 32

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        scheduler.get_num_new_matched_tokens(request, 0)
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1]]
        scheduler.update_state_after_alloc(request, blocks, 32)
        self.assertTrue(scheduler.load_specs["r1"].can_load)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_update_state_after_alloc_zero_external(self, mock_client_cls):
        config = self._make_config(block_size=16)
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        scheduler.load_specs["r1"] = LoadSpec(0, 32, can_load=False)

        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertFalse(scheduler.load_specs["r1"].can_load)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_consumer_no_put(self, mock_client_cls):
        config = self._make_config(kv_role="kv_consumer")
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.request_id = "r1"
        result = scheduler.request_finished(request, [1, 2, 3])
        self.assertEqual(result, (False, None))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_no_tracker(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.request_id = "r1"
        # No tracker means nothing was saved, so there is nothing to send
        # asynchronously: free immediately => (False, None).
        result = scheduler.request_finished(request, [1, 2])
        self.assertEqual(result, (False, None))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_with_saved_tokens(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished(request, [1, 2])
        self.assertTrue(delay)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_request_finished_empty_blocks(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished(request, [])
        self.assertFalse(delay)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_async(self, mock_client_cls):
        config = self._make_config(extra_config={"load_async": True})
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, is_async = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(need, 48)
        self.assertTrue(is_async)


class TestKVPoolSchedulerBuildMeta(unittest.TestCase):
    def _make_config(self, kv_role="kv_producer", block_size=16):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = block_size
        config.cache_config.hash_block_size = block_size
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        return config

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_new_req(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        # Setup a request via update_state_after_alloc
        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        request.num_tokens = 32
        request.num_computed_tokens = 0
        request.block_hashes = [b"h0", b"h1"]
        request.all_token_ids = list(range(32))
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1]]
        scheduler.update_state_after_alloc(request, blocks, 0)

        # Create scheduler output
        new_req_data = MagicMock()
        new_req_data.req_id = "r1"
        new_req_data.num_computed_tokens = 0
        new_req_data.block_ids = [0, 1]
        new_req_data.prompt_token_ids = list(range(32))

        sched_output = MagicMock()
        sched_output.finished_req_ids = set()
        sched_output.preempted_req_ids = set()
        sched_output.scheduled_new_reqs = [new_req_data]
        sched_output.num_scheduled_tokens = {"r1": 32}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        meta = scheduler.build_connector_meta(sched_output)
        self.assertTrue(len(meta.requests) >= 1)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_finished_req(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [0, 1])
        scheduler._unfinished_request_ids.add("r1")

        sched_output = MagicMock()
        sched_output.finished_req_ids = {"r1"}
        sched_output.preempted_req_ids = set()
        sched_output.scheduled_new_reqs = []
        sched_output.num_scheduled_tokens = {}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        _meta = scheduler.build_connector_meta(sched_output)
        self.assertNotIn("r1", scheduler._request_trackers)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_consumer_skip_save(self, mock_client_cls):
        config = self._make_config(kv_role="kv_consumer")
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)

        new_req_data = MagicMock()
        new_req_data.req_id = "r1"
        new_req_data.num_computed_tokens = 0
        new_req_data.block_ids = [0, 1]
        new_req_data.prompt_token_ids = list(range(32))

        sched_output = MagicMock()
        sched_output.finished_req_ids = set()
        sched_output.preempted_req_ids = set()
        sched_output.scheduled_new_reqs = [new_req_data]
        sched_output.num_scheduled_tokens = {"r1": 32}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        _meta = scheduler.build_connector_meta(sched_output)
        # Consumer with no consumer_is_to_put => force_skip_save

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_preempted(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [0, 1])

        sched_output = MagicMock()
        sched_output.finished_req_ids = set()
        sched_output.preempted_req_ids = {"r1"}
        sched_output.scheduled_new_reqs = []
        sched_output.num_scheduled_tokens = {}
        sched_output.scheduled_cached_reqs = MagicMock()
        sched_output.scheduled_cached_reqs.req_ids = []

        _meta = scheduler.build_connector_meta(sched_output)
        self.assertNotIn("r1", scheduler._request_trackers)


class TestLookupKeyClient(unittest.TestCase):
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_lookup(self, mock_zmq, mock_make_socket):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}

        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        mock_socket.recv.return_value = (32).to_bytes(4, "big")

        client = LookupKeyClient(config)
        result = client.lookup(64, [b"\xaa\xbb"])
        self.assertEqual(result, 32)
        mock_socket.send_multipart.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_close(self, mock_zmq, mock_make_socket):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}

        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket

        client = LookupKeyClient(config)
        client.close()
        mock_socket.close.assert_called_once_with(linger=0)


class TestKVPoolSchedulerGenerateKeys(unittest.TestCase):
    """Test generate_keys method."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        return KVPoolScheduler(config, use_layerwise=False)

    def test_generate_keys_basic(self):
        scheduler = self._make_scheduler()
        block_hashes = [b"\xaa\xbb", b"\xcc\xdd"]
        keys, last_key = scheduler.generate_keys(block_hashes)
        self.assertEqual(len(keys), 2)
        self.assertIsNone(last_key)
        self.assertIn("aabb", keys[0])
        self.assertIn("ccdd", keys[1])

    def test_generate_keys_with_last_block(self):
        scheduler = self._make_scheduler()
        block_hashes = [b"\xaa\xbb"]
        keys, last_key = scheduler.generate_keys(block_hashes, req_id="r1", has_last_block=True)
        self.assertEqual(len(keys), 1)
        self.assertIsNotNone(last_key)
        self.assertIn("r1_lastblock", last_key)

    def test_generate_keys_empty(self):
        scheduler = self._make_scheduler()
        keys, last_key = scheduler.generate_keys([])
        self.assertEqual(keys, [])
        self.assertIsNone(last_key)


class TestKVPoolSchedulerStoreQueryKeys(unittest.TestCase):
    """Test _generate_store_query_keys method."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        return KVPoolScheduler(config, use_layerwise=False)

    def test_generate_store_query_keys_basic(self):
        scheduler = self._make_scheduler()
        result = scheduler._generate_store_query_keys([b"\xaa\xbb"])
        # 1 block * 1 tp_rank * 1 pp_rank * 1 pcp * 1 dcp = 1 key per block
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)

    def test_generate_store_query_keys_include_layers(self):
        scheduler = self._make_scheduler()
        scheduler.num_layers = 4
        result = scheduler._generate_store_query_keys([b"\xaa\xbb"], include_layers=True)
        # 1 block * 4 layers = 4 keys per block
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 4)

    def test_generate_store_query_keys_multi_tp(self):
        scheduler = self._make_scheduler()
        scheduler.tp_size = 2
        scheduler.put_step = 1
        result = scheduler._generate_store_query_keys([b"\xaa\xbb"])
        # 1 block * 2 tp_ranks * 1 pp * 1 pcp * 1 dcp = 2 keys
        self.assertEqual(len(result[0]), 2)


class TestKVPoolSchedulerGetStoreLookupHitTokens(unittest.TestCase):
    """Test _get_store_lookup_hit_tokens method."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        return KVPoolScheduler(config, use_layerwise=False)

    def test_all_blocks_hit(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_is_exist.return_value = [1, 1, 1, 1]
        request = MagicMock()
        request.block_hashes = [b"\xaa"] * 4
        result = scheduler._get_store_lookup_hit_tokens(request, 64, 0)
        self.assertEqual(result, 64)

    def test_partial_hit(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_is_exist.return_value = [1, 0, 0, 0]
        request = MagicMock()
        request.block_hashes = [b"\xaa"] * 4
        result = scheduler._get_store_lookup_hit_tokens(request, 64, 0)
        self.assertEqual(result, 16)

    def test_no_hit(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_is_exist.return_value = [0, 0, 0, 0]
        request = MagicMock()
        request.block_hashes = [b"\xaa"] * 4
        result = scheduler._get_store_lookup_hit_tokens(request, 64, 0)
        self.assertEqual(result, 0)

    def test_empty_block_hashes(self):
        scheduler = self._make_scheduler()
        request = MagicMock()
        request.block_hashes = []
        result = scheduler._get_store_lookup_hit_tokens(request, 64, 0)
        self.assertEqual(result, 0)

    def test_with_computed_tokens(self):
        scheduler = self._make_scheduler()
        # 4 blocks, computed 2 blocks -> query blocks 2,3
        scheduler.store_scheduler.batch_is_exist.return_value = [1, 1]
        request = MagicMock()
        request.block_hashes = [b"\xaa"] * 4
        result = scheduler._get_store_lookup_hit_tokens(request, 64, 32)
        self.assertEqual(result, 64)


class TestKVPoolSchedulerStaticMethods(unittest.TestCase):
    """Test static helper methods."""

    def test_uses_hybrid_kv_cache_none(self):
        self.assertFalse(KVPoolScheduler._uses_hybrid_kv_cache(MagicMock(), None))

    def test_uses_hybrid_kv_cache_disabled(self):
        vllm_config = MagicMock()
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_groups = [MagicMock()]
        self.assertFalse(KVPoolScheduler._uses_hybrid_kv_cache(vllm_config, kv_cache_config))

    def test_get_group_family_out_of_range(self):
        self.assertEqual(KVPoolScheduler._get_group_family(None, ["a"], 5), "default")

    def test_get_group_family_valid(self):
        self.assertEqual(KVPoolScheduler._get_group_family(None, ["a", "b"], 1), "b")

    def test_get_group_block_size_out_of_range(self):
        scheduler_mock = MagicMock()
        scheduler_mock.grouped_block_size = [16, 32]
        # Call unbound
        result = KVPoolScheduler._get_group_block_size(scheduler_mock, 5)
        self.assertEqual(result, 16)

    def test_get_group_block_size_valid(self):
        scheduler_mock = MagicMock()
        scheduler_mock.grouped_block_size = [16, 32]
        result = KVPoolScheduler._get_group_block_size(scheduler_mock, 1)
        self.assertEqual(result, 32)


class TestKVPoolSchedulerFloorGranularity(unittest.TestCase):
    """Test _floor_to_cache_transfer_granularity."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_floor(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        scheduler.cache_transfer_granularity = 16
        self.assertEqual(scheduler._floor_to_cache_transfer_granularity(33), 32)
        self.assertEqual(scheduler._floor_to_cache_transfer_granularity(16), 16)
        self.assertEqual(scheduler._floor_to_cache_transfer_granularity(15), 0)


class TestKVPoolSchedulerGetSwClippedBlocks(unittest.TestCase):
    """Test get_sw_clipped_blocks."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_no_swa_blocks(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        scheduler.num_swa_blocks = [0]
        result = scheduler.get_sw_clipped_blocks([[1, 2, 3]])
        self.assertEqual(result, [[1, 2, 3]])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_with_swa_blocks(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        # SWA clipping only runs in hybrid mode; enable it to exercise the path.
        scheduler.use_hybrid = True
        scheduler.num_swa_blocks = [2]
        result = scheduler.get_sw_clipped_blocks([[1, 2, 3, 4, 5]])
        self.assertEqual(result, [[4, 5]])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_empty_blocks(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        scheduler.num_swa_blocks = [0]
        result = scheduler.get_sw_clipped_blocks([])
        self.assertEqual(result, [])


class TestKVPoolSchedulerGetSendingEventId(unittest.TestCase):
    """Test get_sending_event_id."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_increments(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        id1 = scheduler.get_sending_event_id()
        id2 = scheduler.get_sending_event_id()
        self.assertEqual(id1, 0)
        self.assertEqual(id2, 1)


class TestKVPoolSchedulerUpdateFinished(unittest.TestCase):
    """Test update_finished_sending and update_finished_recving."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        return KVPoolScheduler(config, use_layerwise=False)

    def test_update_finished_sending(self):
        scheduler = self._make_scheduler()
        scheduler._delayed_free_req_ids = {"r1", "r2", "r3"}
        scheduler.update_finished_sending({"r1", "r2"})
        self.assertEqual(scheduler._delayed_free_req_ids, {"r3"})

    def test_update_finished_sending_none(self):
        scheduler = self._make_scheduler()
        scheduler._delayed_free_req_ids = {"r1"}
        scheduler.update_finished_sending(None)
        self.assertEqual(scheduler._delayed_free_req_ids, {"r1"})

    def test_update_finished_recving(self):
        scheduler = self._make_scheduler()
        scheduler._loading_req_ids = {"r1", "r2"}
        scheduler.update_finished_recving({"r1"})
        self.assertEqual(scheduler._loading_req_ids, {"r2"})

    def test_update_finished_recving_none(self):
        scheduler = self._make_scheduler()
        scheduler._loading_req_ids = {"r1"}
        scheduler.update_finished_recving(None)
        self.assertEqual(scheduler._loading_req_ids, {"r1"})


class TestKVPoolSchedulerBindBlockPool(unittest.TestCase):
    """Test bind_gpu_block_pool."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_bind(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        self.assertIsNone(scheduler._block_pool)
        mock_pool = MagicMock()
        scheduler.bind_gpu_block_pool(mock_pool)
        self.assertIs(scheduler._block_pool, mock_pool)


class TestKVPoolSchedulerUpdateConnectorOutput(unittest.TestCase):
    """Test update_connector_output."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        config.parallel_config.world_size = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        scheduler._block_pool = MagicMock()
        return scheduler

    def test_completed_event_frees_blocks(self):
        scheduler = self._make_scheduler()
        scheduler.sending_events = {1: 1}  # already 1 worker completed
        scheduler.sending_blocks = {1: [10, 20, 30]}
        scheduler._expected_worker_count = 2

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
            AscendStoreKVConnectorWorkerMetadata,
        )

        meta = AscendStoreKVConnectorWorkerMetadata({1: 1})
        output = MagicMock()
        output.kv_connector_worker_meta = meta
        scheduler.update_connector_output(output)
        # total = 1 + 1 = 2 >= 2 => free blocks
        scheduler._block_pool.free_blocks.assert_called_once()
        self.assertNotIn(1, scheduler.sending_blocks)
        self.assertNotIn(1, scheduler.sending_events)

    def test_incomplete_event_keeps_blocks(self):
        scheduler = self._make_scheduler()
        scheduler.sending_events = {1: 0}
        scheduler.sending_blocks = {1: [10, 20]}
        scheduler._expected_worker_count = 2

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
            AscendStoreKVConnectorWorkerMetadata,
        )

        meta = AscendStoreKVConnectorWorkerMetadata({1: 1})
        output = MagicMock()
        output.kv_connector_worker_meta = meta
        scheduler.update_connector_output(output)
        # total = 0 + 1 = 1 < 2 => keep blocks
        scheduler._block_pool.free_blocks.assert_not_called()
        self.assertEqual(scheduler.sending_events[1], 1)

    def test_invalid_event_id(self):
        scheduler = self._make_scheduler()
        scheduler.sending_events = {}
        scheduler.sending_blocks = {}

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
            AscendStoreKVConnectorWorkerMetadata,
        )

        meta = AscendStoreKVConnectorWorkerMetadata({99: 1})
        output = MagicMock()
        output.kv_connector_worker_meta = meta
        scheduler.update_connector_output(output)
        # No crash, no free
        scheduler._block_pool.free_blocks.assert_not_called()

    def test_non_ascend_meta_ignored(self):
        scheduler = self._make_scheduler()
        output = MagicMock()
        output.kv_connector_worker_meta = MagicMock()  # Not AscendStoreKVConnectorWorkerMetadata
        scheduler.update_connector_output(output)
        scheduler._block_pool.free_blocks.assert_not_called()


class TestKVPoolSchedulerRequestFinishedAllGroups(unittest.TestCase):
    """Test request_finished_all_groups."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls, kv_role="kv_producer"):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        scheduler.num_swa_blocks = [0]
        return scheduler

    def test_consumer_no_put(self):
        scheduler = self._make_scheduler(kv_role="kv_consumer")
        request = MagicMock()
        request.request_id = "r1"
        delay, extra = scheduler.request_finished_all_groups(request, ([1, 2],))
        self.assertFalse(delay)

    def test_no_tracker(self):
        scheduler = self._make_scheduler()
        request = MagicMock()
        request.request_id = "r_nonexist"
        delay, _ = scheduler.request_finished_all_groups(request, ([1, 2],))
        self.assertTrue(delay)

    def test_tracker_not_saved(self):
        scheduler = self._make_scheduler()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker("r1", 32, allocated_block_ids=[0, 1], num_saved_tokens=0)
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished_all_groups(request, ([1, 2],))
        self.assertFalse(delay)

    def test_delay_free_with_blocks(self):
        scheduler = self._make_scheduler()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker("r1", 32, allocated_block_ids=[0, 1], num_saved_tokens=32)
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished_all_groups(request, ([1, 2],))
        self.assertTrue(delay)
        self.assertIn("r1", scheduler._delayed_free_req_ids)

    def test_no_delay_empty_blocks(self):
        scheduler = self._make_scheduler()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker("r1", 32, allocated_block_ids=[0, 1], num_saved_tokens=32)
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished_all_groups(request, ([],))
        self.assertFalse(delay)


class TestKVPoolSchedulerInferMambaGroups(unittest.TestCase):
    """Test _infer_mamba_groups."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_no_config(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        self.assertEqual(scheduler._infer_mamba_groups(), [])


class TestKVPoolSchedulerGetLayerwiseGvaHitTokens(unittest.TestCase):
    """Test _get_layerwise_gva_hit_tokens."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        # Concrete model_config values so KVPoolScheduler.__init__ int math
        # (num_kv_head < tp_size, get_num_layers, model name split, ...) works.
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        return scheduler

    def test_all_hit(self):
        scheduler = self._make_scheduler()
        key_info = MagicMock()
        key_info.size.return_value = 1
        key_info.gva_list.return_value = [0x1000]
        scheduler.store_scheduler.batch_get_key_info.return_value = [key_info, key_info]

        request = MagicMock()
        request.block_hashes = [b"\xaa", b"\xbb"]
        result = scheduler._get_layerwise_gva_hit_tokens(request, 32, 0)
        self.assertEqual(result, 32)

    def test_partial_hit(self):
        scheduler = self._make_scheduler()
        hit_info = MagicMock()
        hit_info.size.return_value = 1
        hit_info.gva_list.return_value = [0x1000]
        miss_info = MagicMock()
        miss_info.size.return_value = 0
        scheduler.store_scheduler.batch_get_key_info.return_value = [hit_info, miss_info]

        request = MagicMock()
        request.block_hashes = [b"\xaa", b"\xbb"]
        result = scheduler._get_layerwise_gva_hit_tokens(request, 32, 0)
        self.assertEqual(result, 16)

    def test_no_hit(self):
        scheduler = self._make_scheduler()
        miss_info = MagicMock()
        miss_info.size.return_value = 0
        scheduler.store_scheduler.batch_get_key_info.return_value = [miss_info]

        request = MagicMock()
        request.block_hashes = [b"\xaa"]
        result = scheduler._get_layerwise_gva_hit_tokens(request, 16, 0)
        self.assertEqual(result, 0)

    def test_with_computed_tokens(self):
        scheduler = self._make_scheduler()
        hit_info = MagicMock()
        hit_info.size.return_value = 1
        hit_info.gva_list.return_value = [0x1000]
        scheduler.store_scheduler.batch_get_key_info.return_value = [hit_info, hit_info]

        request = MagicMock()
        request.block_hashes = [b"\xaa", b"\xbb", b"\xcc", b"\xdd"]
        result = scheduler._get_layerwise_gva_hit_tokens(request, 64, 32)
        self.assertEqual(result, 64)


class TestKVPoolSchedulerUpdateStateAfterAllocBranches(unittest.TestCase):
    """Test update_state_after_alloc additional branches."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls, extra_config=None):
        config = MagicMock()
        config.kv_transfer_config.kv_role = "kv_producer"
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        config.kv_transfer_config.get_from_extra_config.return_value = True
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.cache_config.block_size = 16
        config.cache_config.hash_block_size = 16
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.rank = 0
        config.parallel_config.world_size = 1
        config.model_config.model = "org/llama-7b"
        config.model_config.use_mla = False
        config.model_config.hf_text_config = MagicMock(spec=[])
        config.model_config.get_total_num_kv_heads.return_value = 1
        config.model_config.get_num_layers.return_value = 2
        return KVPoolScheduler(config, use_layerwise=False)

    def test_async_adds_loading_req(self):
        scheduler = self._make_scheduler(extra_config={"load_async": True})
        scheduler.load_specs["r1"] = LoadSpec(0, 32, can_load=True)

        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1]]
        scheduler.update_state_after_alloc(request, blocks, 32)
        self.assertIn("r1", scheduler._loading_req_ids)

    def test_no_load_spec_returns_early(self):
        scheduler = self._make_scheduler()
        request = MagicMock()
        request.request_id = "r_noexist"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertNotIn("r_noexist", scheduler.load_specs)

    def test_zero_external_tokens_layerwise(self):
        scheduler = self._make_scheduler()
        scheduler.use_layerwise = True
        scheduler.load_specs["r1"] = LoadSpec(0, 32, can_load=False)

        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        # layerwise + kvpool_cached > 0 => can_load = True
        self.assertTrue(scheduler.load_specs["r1"].can_load)


if __name__ == "__main__":
    unittest.main()
