import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    get_zmq_rpc_path_lookup,
)


class TestGetZmqRpcPathLookup(unittest.TestCase):
    def _make_vllm_config(self, dp_rank=0, extra_config=None):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = dp_rank
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        return config

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs")
    def test_default_port(self, mock_envs):
        mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
        config = self._make_vllm_config(dp_rank=0)
        result = get_zmq_rpc_path_lookup(config)
        self.assertEqual(result, "ipc:///tmp/vllm/lookup_rpc_port_0_dp_rank0")

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs")
    def test_lookup_rpc_port(self, mock_envs):
        mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
        config = self._make_vllm_config(
            dp_rank=1, extra_config={"lookup_rpc_port": 5000}
        )
        result = get_zmq_rpc_path_lookup(config)
        self.assertEqual(result, "ipc:///tmp/vllm/lookup_rpc_port_5000_dp_rank1")

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs")
    def test_mooncake_rpc_port_fallback(self, mock_envs):
        mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
        config = self._make_vllm_config(
            dp_rank=0, extra_config={"mooncake_rpc_port": 6000}
        )
        result = get_zmq_rpc_path_lookup(config)
        self.assertEqual(result, "ipc:///tmp/vllm/lookup_rpc_port_6000_dp_rank0")

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs")
    def test_lookup_rpc_port_takes_priority(self, mock_envs):
        mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
        config = self._make_vllm_config(
            dp_rank=0,
            extra_config={"lookup_rpc_port": 5000, "mooncake_rpc_port": 6000},
        )
        result = get_zmq_rpc_path_lookup(config)
        self.assertIn("5000", result)


class TestKVPoolScheduler(unittest.TestCase):
    def _make_scheduler(
        self,
        kv_role="kv_producer",
        block_size=8,
        consumer_is_to_load=False,
        consumer_is_to_put=False,
        discard_partial_chunks=True,
        load_async=False,
    ):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
            KVPoolScheduler,
        )

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config.kv_role = kv_role
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "consumer_is_to_load": consumer_is_to_load,
            "consumer_is_to_put": consumer_is_to_put,
            "load_async": load_async,
        }
        vllm_config.kv_transfer_config.get_from_extra_config.return_value = discard_partial_chunks
        vllm_config.cache_config.block_size = block_size
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1

        # Patch the LookupKeyClient
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            scheduler = KVPoolScheduler(vllm_config, use_layerwise=False)

        scheduler.client = mock_client
        return scheduler

    def test_get_num_new_matched_tokens_consumer_no_load(self):
        scheduler = self._make_scheduler(kv_role="kv_consumer", consumer_is_to_load=False)
        request = MagicMock()
        request.prompt_token_ids = list(range(32))
        result = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(result, (0, False))

    def test_get_num_new_matched_tokens_short_prompt(self):
        scheduler = self._make_scheduler(block_size=16)
        request = MagicMock()
        request.prompt_token_ids = list(range(8))  # shorter than block_size
        result = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(result, (0, False))

    def test_get_num_new_matched_tokens_hit(self):
        scheduler = self._make_scheduler(block_size=8)
        scheduler.client.lookup.return_value = 16

        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        request.num_tokens = 32
        request.block_hashes = [b"h0", b"h1", b"h2", b"h3"]

        num_new, is_async = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(num_new, 16)
        self.assertFalse(is_async)
        self.assertIn("r1", scheduler.load_specs)

    def test_get_num_new_matched_tokens_all_hit_minus_one(self):
        scheduler = self._make_scheduler(block_size=8)
        scheduler.client.lookup.return_value = 32  # equals num_tokens

        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        request.num_tokens = 32
        request.block_hashes = [b"h0", b"h1", b"h2", b"h3"]

        num_new, _ = scheduler.get_num_new_matched_tokens(request, 0)
        # external hit = 32, but 32 == num_tokens so becomes 31
        # need_to_allocate = 31 - 0 = 31
        self.assertEqual(num_new, 31)

    def test_get_num_new_matched_tokens_no_new_tokens(self):
        scheduler = self._make_scheduler(block_size=8)
        scheduler.client.lookup.return_value = 8

        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        request.num_tokens = 32
        request.block_hashes = [b"h0", b"h1", b"h2", b"h3"]

        num_new, _ = scheduler.get_num_new_matched_tokens(request, 16)
        # external hit = 8, computed = 16, 8 < 16 -> 0
        self.assertEqual(num_new, 0)

    def test_get_num_new_matched_tokens_async(self):
        scheduler = self._make_scheduler(block_size=8, load_async=True)
        scheduler.client.lookup.return_value = 16
        scheduler.use_layerwise = False

        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(32))
        request.num_tokens = 32
        request.block_hashes = [b"h0", b"h1", b"h2", b"h3"]

        _, is_async = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertTrue(is_async)

    def test_update_state_after_alloc_no_load_spec(self):
        scheduler = self._make_scheduler()
        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertIn("r1", scheduler._unfinished_request_ids)

    def test_update_state_after_alloc_with_external_tokens(self):
        scheduler = self._make_scheduler(block_size=8)
        scheduler.load_specs["r1"] = LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=False
        )
        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1, 2]]
        scheduler.update_state_after_alloc(request, blocks, 16)
        self.assertTrue(scheduler.load_specs["r1"].can_load)

    def test_update_state_after_alloc_zero_external(self):
        scheduler = self._make_scheduler()
        scheduler.load_specs["r1"] = LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=False
        )
        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertFalse(scheduler.load_specs["r1"].can_load)

    def test_request_finished_consumer_no_put(self):
        scheduler = self._make_scheduler(kv_role="kv_consumer", consumer_is_to_put=False)
        request = MagicMock()
        request.request_id = "r1"
        result = scheduler.request_finished(request, [0, 1])
        self.assertEqual(result, (False, None))

    def test_request_finished_no_tracker(self):
        scheduler = self._make_scheduler()
        request = MagicMock()
        request.request_id = "r_none"
        result = scheduler.request_finished(request, [0, 1])
        self.assertEqual(result, (False, None))

    def test_request_finished_with_saved_tokens(self):
        scheduler = self._make_scheduler()
        tracker = RequestTracker("r1", 16, [0, 1], num_saved_tokens=16)
        scheduler._request_trackers["r1"] = tracker
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished(request, [0, 1])
        self.assertTrue(delay)

    def test_request_finished_no_saved_tokens(self):
        scheduler = self._make_scheduler()
        tracker = RequestTracker("r1", 16, [0, 1], num_saved_tokens=0)
        scheduler._request_trackers["r1"] = tracker
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished(request, [0, 1])
        self.assertFalse(delay)

    def test_request_finished_empty_block_ids(self):
        scheduler = self._make_scheduler()
        tracker = RequestTracker("r1", 16, [0, 1], num_saved_tokens=16)
        scheduler._request_trackers["r1"] = tracker
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished(request, [])
        self.assertFalse(delay)


class TestBuildConnectorMeta(unittest.TestCase):
    def _make_scheduler(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
            KVPoolScheduler,
        )

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config.kv_role = "kv_producer"
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "consumer_is_to_load": False,
            "consumer_is_to_put": False,
            "load_async": False,
        }
        vllm_config.kv_transfer_config.get_from_extra_config.return_value = True
        vllm_config.cache_config.block_size = 8
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
        ):
            scheduler = KVPoolScheduler(vllm_config, use_layerwise=False)
        return scheduler

    def test_build_connector_meta_new_req(self):
        scheduler = self._make_scheduler()

        # Setup a request in unfinished_requests
        request_obj = MagicMock()
        request_obj.request_id = "r1"
        request_obj.prompt_token_ids = list(range(16))
        request_obj.block_hashes = [b"h0", b"h1"]
        scheduler._unfinished_requests["r1"] = (request_obj, [])

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.preempted_req_ids = set()

        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.num_computed_tokens = 0
        new_req.block_ids = [0, 1]
        new_req.prompt_token_ids = list(range(16))
        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.num_scheduled_tokens = {"r1": 16}
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []

        meta = scheduler.build_connector_meta(scheduler_output)
        self.assertIsInstance(meta, AscendConnectorMetadata)

    def test_build_connector_meta_finished_req(self):
        scheduler = self._make_scheduler()
        scheduler._request_trackers["r1"] = RequestTracker("r1", 16, [0], 0)
        scheduler._unfinished_requests["r1"] = (MagicMock(), [])
        scheduler._unfinished_request_ids.add("r1")

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = {"r1"}
        scheduler_output.preempted_req_ids = set()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []
        scheduler_output.num_scheduled_tokens = {}

        meta = scheduler.build_connector_meta(scheduler_output)
        self.assertNotIn("r1", scheduler._request_trackers)

    def test_build_connector_meta_preempted_req(self):
        scheduler = self._make_scheduler()
        scheduler._request_trackers["r2"] = RequestTracker("r2", 16, [0], 0)
        scheduler._unfinished_requests["r2"] = (MagicMock(), [])

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.preempted_req_ids = {"r2"}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []
        scheduler_output.num_scheduled_tokens = {}

        meta = scheduler.build_connector_meta(scheduler_output)
        self.assertNotIn("r2", scheduler._request_trackers)


if __name__ == "__main__":
    unittest.main()
