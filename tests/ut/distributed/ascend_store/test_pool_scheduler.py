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
    LoadSpec,
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
    LookupKeyClient,
    get_zmq_rpc_path_lookup,
)


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
        config.cache_config.block_size = block_size
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
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import LoadSpec

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
        # No tracker => tracker is None => num_saved_tokens check skipped
        # delay_free_blocks = len(block_ids) > 0 => True
        result = scheduler.request_finished(request, [1, 2])
        # tracker is None so condition `tracker.num_saved_tokens <= 0` is not checked
        # but tracker is None means `tracker is not None` is False => (False, None)
        # Actually: tracker = self._request_trackers.get("r1") => None
        # `if tracker is not None and tracker.num_saved_tokens <= 0:` => False
        # delay_free_blocks = len([1,2]) > 0 => True
        self.assertEqual(result, (True, None))

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
        config.cache_config.block_size = block_size
        return config

    def _make_request(self, req_id="r1", prompt_len=64, num_computed_tokens=0):
        request = MagicMock()
        request.request_id = req_id
        request.req_id = req_id
        request.prompt_token_ids = list(range(prompt_len))
        request.all_token_ids = list(range(prompt_len))
        request.num_tokens = prompt_len
        request.num_computed_tokens = num_computed_tokens
        request.block_hashes = [f"h{i}".encode() for i in range((prompt_len + 15) // 16)]
        return request

    def _make_cached_reqs(self, req_ids=None, new_block_ids=None, num_computed_tokens=None):
        req_ids = [] if req_ids is None else req_ids
        cached_reqs = MagicMock()
        cached_reqs.req_ids = req_ids
        cached_reqs.new_block_ids = new_block_ids if new_block_ids is not None else [[] for _ in req_ids]
        cached_reqs.num_computed_tokens = (
            num_computed_tokens if num_computed_tokens is not None else [0 for _ in req_ids]
        )
        return cached_reqs

    def _make_sched_output(
        self,
        scheduled_new_reqs=None,
        cached_reqs=None,
        num_scheduled_tokens=None,
        finished_req_ids=None,
        preempted_req_ids=None,
    ):
        sched_output = MagicMock()
        sched_output.finished_req_ids = finished_req_ids or set()
        sched_output.preempted_req_ids = preempted_req_ids or set()
        sched_output.scheduled_new_reqs = scheduled_new_reqs or []
        sched_output.num_scheduled_tokens = num_scheduled_tokens or {}
        sched_output.scheduled_cached_reqs = cached_reqs or self._make_cached_reqs()
        return sched_output

    def _seed_cached_request(
        self,
        scheduler,
        req_id="r1",
        token_len=16,
        num_saved_tokens=0,
        block_ids=None,
        block_ids_by_group=None,
        prompt_len=64,
    ):
        request = self._make_request(req_id=req_id, prompt_len=prompt_len, num_computed_tokens=token_len)
        if block_ids_by_group is None:
            block_ids_by_group = [block_ids or [0]]
        tracker = RequestTracker(
            req_id=req_id,
            token_len=token_len,
            allocated_block_ids_by_group=block_ids_by_group,
            num_saved_tokens=num_saved_tokens,
            token_ids=list(range(token_len)),
            block_sizes=[16 for _ in block_ids_by_group],
        )
        scheduler._request_trackers[req_id] = tracker
        scheduler._unfinished_requests[req_id] = (request, block_ids_by_group)
        scheduler._unfinished_request_ids.add(req_id)
        return request, tracker

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

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_cached_decode_request_updates_tracker(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        _, tracker = self._seed_cached_request(scheduler, token_len=16, block_ids=[0])
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[1]], num_computed_tokens=[16])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 16},
        )

        meta = scheduler.build_connector_meta(sched_output)

        self.assertEqual(tracker.token_len, 32)
        self.assertEqual(tracker.allocated_block_ids, [0, 1])
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests[0].req_id, "r1")
        self.assertEqual(meta.requests[0].block_ids, [0, 1])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_cached_chunked_request_generates_meta_at_boundary(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        self._seed_cached_request(scheduler, token_len=16, num_saved_tokens=16, block_ids=[0])
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[1]], num_computed_tokens=[16])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 16},
        )

        meta = scheduler.build_connector_meta(sched_output)

        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests[0].token_len_chunk, 32)
        self.assertTrue(meta.requests[0].can_save)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_cached_chunked_request_skips_before_boundary(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        _, tracker = self._seed_cached_request(scheduler, token_len=16, num_saved_tokens=16, block_ids=[0])
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[1]], num_computed_tokens=[16])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 4},
        )

        meta = scheduler.build_connector_meta(sched_output)

        self.assertEqual(tracker.token_len, 20)
        self.assertEqual(meta.requests, [])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_cached_request_skips_after_prompt_computed(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        _, tracker = self._seed_cached_request(scheduler, token_len=64, block_ids=[0, 1, 2, 3])
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[4]], num_computed_tokens=[64])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 1},
        )

        meta = scheduler.build_connector_meta(sched_output)

        self.assertEqual(meta.requests, [])
        self.assertEqual(tracker.allocated_block_ids, [0, 1, 2, 3])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_cached_request_missing_unfinished_raises(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=16,
            allocated_block_ids=[0],
        )
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[1]], num_computed_tokens=[16])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 16},
        )

        with self.assertRaises(ValueError):
            scheduler.build_connector_meta(sched_output)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_preempted_resumed_rebuilds_tracker(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        self._seed_cached_request(scheduler, token_len=16, block_ids=[0])
        scheduler._preempted_req_ids.add("r1")
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[7]], num_computed_tokens=[16])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 16},
        )

        meta = scheduler.build_connector_meta(sched_output)

        self.assertNotIn("r1", scheduler._preempted_req_ids)
        self.assertEqual(scheduler._request_trackers["r1"].token_len, 32)
        self.assertEqual(scheduler._request_trackers["r1"].allocated_block_ids, [7])
        self.assertEqual(meta.requests[0].block_ids, [7])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_preempted_resumed_consumes_load_spec(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        self._seed_cached_request(scheduler, token_len=16, block_ids=[0])
        scheduler._preempted_req_ids.add("r1")
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True)
        scheduler.load_specs["r1"] = load_spec
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[7]], num_computed_tokens=[16])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 16},
        )

        meta = scheduler.build_connector_meta(sched_output)

        self.assertNotIn("r1", scheduler.load_specs)
        self.assertIs(meta.requests[0].load_spec, load_spec)
        self.assertTrue(meta.requests[0].can_save)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_preempted_resumed_without_load_spec_saves(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        self._seed_cached_request(scheduler, token_len=16, block_ids=[0])
        scheduler._preempted_req_ids.add("r1")
        cached_reqs = self._make_cached_reqs(req_ids=["r1"], new_block_ids=[[7]], num_computed_tokens=[16])
        sched_output = self._make_sched_output(
            cached_reqs=cached_reqs,
            num_scheduled_tokens={"r1": 16},
        )

        meta = scheduler.build_connector_meta(sched_output)

        self.assertEqual(len(meta.requests), 1)
        self.assertIsNone(meta.requests[0].load_spec)
        self.assertTrue(meta.requests[0].can_save)


class TestLookupKeyClient(unittest.TestCase):
    class FakeEncoder:
        def __init__(self):
            self.values = []

        def encode(self, value):
            self.values.append(value)
            return [f"encoded:{value!r}".encode()]

    def _make_config(self, extra_config=None):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        return config

    def _make_client(self, mock_make_socket, recv_value=32):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        mock_socket.recv.return_value = recv_value.to_bytes(4, "big")
        client = LookupKeyClient(self._make_config())
        return client, mock_socket

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_lookup(self, mock_zmq, mock_make_socket):
        client, mock_socket = self._make_client(mock_make_socket)
        result = client.lookup(64, [b"\xaa\xbb"])
        self.assertEqual(result, 32)
        mock_socket.send_multipart.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_lookup_encodes_group_ids_and_hash_frames(self, mock_zmq, mock_make_socket):
        client, mock_socket = self._make_client(mock_make_socket, recv_value=48)
        fake_encoder = self.FakeEncoder()
        client.encoder = fake_encoder

        result = client.lookup(64, [b"\x01", b"\x02"], kv_cache_group_ids=[0, 2])

        self.assertEqual(result, 48)
        self.assertEqual(fake_encoder.values, [[b"\x01".hex(), b"\x02".hex()], [0, 2]])
        mock_socket.send_multipart.assert_called_once_with(
            [
                (64).to_bytes(4, byteorder="big"),
                b"encoded:[0, 2]",
                b"encoded:['01', '02']",
            ],
            copy=False,
        )

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_lookup_send_multipart_timeout_raises(self, mock_zmq, mock_make_socket):
        client, mock_socket = self._make_client(mock_make_socket)
        mock_socket.send_multipart.side_effect = TimeoutError("send timeout")

        with self.assertRaises(TimeoutError):
            client.lookup(64, [b"\xaa\xbb"])

        mock_socket.recv.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_lookup_recv_timeout_raises(self, mock_zmq, mock_make_socket):
        client, mock_socket = self._make_client(mock_make_socket)
        mock_socket.recv.side_effect = TimeoutError("recv timeout")

        with self.assertRaises(TimeoutError):
            client.lookup(64, [b"\xaa\xbb"])

        mock_socket.send_multipart.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq")
    def test_close(self, mock_zmq, mock_make_socket):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket

        client = LookupKeyClient(self._make_config())
        client.close()
        mock_socket.close.assert_called_once_with(linger=0)


if __name__ == "__main__":
    unittest.main()
