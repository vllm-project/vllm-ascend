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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LoadSpec,
    RequestTracker,
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


def make_config(kv_role="kv_producer", extra_config=None, block_size=16):
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
    config.model_config.model = "org/llama-7b"
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 2
    return config


class TestGetZmqRpcPathLookup(unittest.TestCase):
    def test_rpc_path(self):
        cases = [({}, 0, 0), ({"lookup_rpc_port": 5555}, 1, 5555), ({"mooncake_rpc_port": 6666}, 0, 6666)]
        for extra_config, rank, port in cases:
            with self.subTest(extra_config=extra_config, rank=rank):
                config = MagicMock()
                config.parallel_config.data_parallel_rank = rank
                config.kv_transfer_config.kv_connector_extra_config = extra_config
                result = get_zmq_rpc_path_lookup(config)
                self.assertIn(f"lookup_rpc_port_{port}", result)
                self.assertIn(f"dp_rank{rank}", result)


class TestKVPoolScheduler(unittest.TestCase):
    def _make_config(self, kv_role="kv_producer", extra_config=None, block_size=16):
        return make_config(kv_role, extra_config, block_size)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_early_returns(self, mock_client_cls):
        for role, block_size, token_count in [("kv_consumer", 16, 64), ("kv_producer", 64, 32)]:
            with self.subTest(role=role, block_size=block_size):
                scheduler = KVPoolScheduler(self._make_config(role, block_size=block_size), use_layerwise=False)
                request = MagicMock(prompt_token_ids=list(range(token_count)))
                self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_hit(self, mock_client_cls):
        request = MagicMock(
            prompt_token_ids=list(range(64)),
            num_tokens=64,
            request_id="r1",
            block_hashes=[b"h"] * 4,
        )
        cases = [
            (48, 16, False, 32),
            (48, 0, True, 48),
            (64, 0, False, 63),
            (16, 32, False, 0),
        ]
        for lookup_hit, computed, load_async, expected in cases:
            with self.subTest(lookup_hit=lookup_hit, computed=computed, load_async=load_async):
                mock_client_cls.reset_mock()
                mock_client_cls.return_value.lookup.return_value = lookup_hit
                config = self._make_config(extra_config={"load_async": True} if load_async else None)
                scheduler = KVPoolScheduler(config, use_layerwise=False)
                need, is_async = scheduler.get_num_new_matched_tokens(request, computed)
                self.assertEqual((need, is_async), (expected, load_async and expected > 0))
                self.assertEqual("r1" in scheduler.load_specs, expected > 0)

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
        self.assertEqual(scheduler.load_specs["r1"].kvpool_cached_tokens, 63)
        self.assertEqual(scheduler.load_specs["r1"].kvpool_store_skip_tokens, 64)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_layerwise_mtp_hit_uses_safe_load_extent(self, mock_client_cls):
        scheduler = KVPoolScheduler(
            self._make_config(block_size=16, extra_config={"backend": "memcache"}),
            use_layerwise=True,
        )
        scheduler.use_eagle = True
        scheduler.cache_transfer_granularity = 16
        scheduler._get_layerwise_hit_tokens = MagicMock(return_value=64)

        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        need, is_async = scheduler.get_num_new_matched_tokens(request, 48)

        self.assertEqual(need, 0)
        self.assertFalse(is_async)
        load_spec = scheduler.load_specs["r1"]
        self.assertEqual(load_spec.kvpool_cached_tokens, 48)
        self.assertEqual(load_spec.kvpool_store_skip_tokens, 64)
        self.assertTrue(load_spec.can_load)
        scheduler.update_state_after_alloc(request, MagicMock(), 0)
        self.assertTrue(load_spec.can_load)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_layerwise_mtp_hit_on_final_block_boundary_not_trimmed(self, mock_client_cls):
        scheduler = KVPoolScheduler(
            self._make_config(block_size=16, extra_config={"backend": "memcache"}),
            use_layerwise=True,
        )
        scheduler.use_eagle = True
        scheduler.cache_transfer_granularity = 16
        scheduler._get_layerwise_hit_tokens = MagicMock(return_value=96)

        request = MagicMock()
        request.prompt_token_ids = list(range(100))
        request.num_tokens = 101
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 7

        need, is_async = scheduler.get_num_new_matched_tokens(request, 0)

        # The hit (96) stops exactly on the final block's start boundary
        # ((101 - 1) // 16 * 16 = 96) and must be loaded as-is instead of
        # being trimmed back by one whole lcm block.
        self.assertEqual(need, 96)
        self.assertFalse(is_async)
        load_spec = scheduler.load_specs["r1"]
        self.assertEqual(load_spec.kvpool_cached_tokens, 96)
        self.assertEqual(load_spec.kvpool_store_skip_tokens, 96)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_get_num_new_matched_tokens_full_hbm_hit_skips_external_lookup(self, mock_client_cls):
        scheduler = KVPoolScheduler(self._make_config(block_size=16), use_layerwise=False)
        request = MagicMock()
        request.prompt_token_ids = list(range(64))
        request.num_tokens = 64
        request.request_id = "r1"
        request.block_hashes = [b"h"] * 4

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 64), (0, False))
        mock_client_cls.return_value.lookup.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_update_state_after_alloc_no_load_spec(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertIn("r1", scheduler._unfinished_requests)

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
    def test_request_finished_early_returns(self, mock_client_cls):
        for role in ("kv_consumer", "kv_producer"):
            with self.subTest(role=role):
                scheduler = KVPoolScheduler(self._make_config(kv_role=role), use_layerwise=False)
                result = scheduler.request_finished(MagicMock(request_id="r1"), [1, 2])
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
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import RequestTracker

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
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import RequestTracker

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
    def _make_config(self, kv_role="kv_producer", block_size=16, extra_config=None, num_layers=2):
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
        config.model_config.get_num_layers.return_value = num_layers
        return config

    def _set_running_chunk(self, scheduler):
        request = MagicMock()
        request.num_computed_tokens = 16
        request.num_prompt_tokens = 64
        request.prompt_token_ids = list(range(64))
        request.all_token_ids = list(range(64))
        request.block_hashes = [b"h0", b"h1", b"h2", b"h3"]
        scheduler._unfinished_requests["r1"] = (request, [[0]])
        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=16,
            allocated_block_ids=[0],
            num_saved_tokens=16,
            token_ids=list(range(16)),
            num_prompt_tokens=64,
        )

    def _make_running_chunk_output(self, new_block_ids):
        sched_output = MagicMock()
        sched_output.finished_req_ids = set()
        sched_output.preempted_req_ids = set()
        sched_output.scheduled_new_reqs = []
        sched_output.num_scheduled_tokens = {"r1": 16}
        sched_output.scheduled_cached_reqs.req_ids = ["r1"]
        sched_output.scheduled_cached_reqs.new_block_ids = [new_block_ids]
        return sched_output

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_running_chunk_passes_computed_tokens_to_tracker(self, mock_client_cls):
        scheduler = KVPoolScheduler(self._make_config(), use_layerwise=False)
        request = MagicMock()
        request.num_computed_tokens = 128
        request.num_prompt_tokens = 256
        request.prompt_token_ids = list(range(256))
        request.all_token_ids = list(range(256))
        request.block_hashes = [b"h"] * 16
        scheduler._unfinished_requests["r1"] = (request, [[] for _ in range(4)])
        request_tracker = RequestTracker(
            req_id="r1",
            token_len=128,
            allocated_block_ids_by_group=[[] for _ in range(4)],
        )
        request_tracker.update = MagicMock()
        scheduler._request_trackers["r1"] = request_tracker
        new_block_ids = (
            [21, 22, 23, 24, 25, 26, 27, 28],
            [0, 0, 0, 0, 10, 11, 12, 29],
            [0, 0, 0, 0, 14, 15, 16, 30],
            [0, 0, 0, 0, 18, 19, 20, 31],
        )
        scheduler._build_req_meta = MagicMock(return_value=None)

        scheduler._process_running_cached_request(
            new_block_ids,
            "r1",
            0,
            MagicMock(),
            self._make_running_chunk_output(new_block_ids),
            False,
        )

        request_tracker.update.assert_called_once_with(new_block_ids, 128)

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
    def test_running_chunk_reloads_prefix_with_layer_reuse(self, mock_client_cls):
        config = self._make_config(
            extra_config={
                "backend": "memcache",
                "layerwise_num_shared_buffers": 1,
            },
            num_layers=4,
        )
        scheduler = KVPoolScheduler(config, use_layerwise=True)
        self._set_running_chunk(scheduler)

        meta = scheduler.build_connector_meta(self._make_running_chunk_output([]))

        self.assertTrue(scheduler.layerwise_offload)
        self.assertEqual(len(meta.requests), 1)
        load_spec = meta.requests[0].load_spec
        self.assertIsNotNone(load_spec)
        self.assertEqual(load_spec.vllm_cached_tokens, 16)
        self.assertEqual(load_spec.kvpool_cached_tokens, 16)
        self.assertTrue(load_spec.can_load)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_running_decode_preserves_partial_block_with_layer_reuse(self, mock_client_cls):
        config = self._make_config(
            extra_config={
                "backend": "memcache",
                "layerwise_num_shared_buffers": 1,
            },
            num_layers=4,
        )
        scheduler = KVPoolScheduler(config, use_layerwise=True)
        request = MagicMock()
        request.num_computed_tokens = 32
        request.num_prompt_tokens = 32
        request.prompt_token_ids = list(range(32))
        request.all_token_ids = list(range(33))
        request.block_hashes = [b"h0", b"h1"]
        scheduler._unfinished_requests["r1"] = (request, [[0, 1, 2]])
        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1, 2],
            num_saved_tokens=32,
            token_ids=list(range(32)),
            num_prompt_tokens=32,
        )
        sched_output = self._make_running_chunk_output([])
        sched_output.num_scheduled_tokens = {"r1": 1}

        meta = scheduler.build_connector_meta(sched_output)

        self.assertEqual(len(meta.requests), 1)
        request_meta = meta.requests[0]
        self.assertTrue(request_meta.can_save)
        self.assertEqual(request_meta.save_start_token, 32)
        self.assertEqual(request_meta.save_end_token, 32)
        self.assertEqual(request_meta.target_token_len, 33)
        self.assertIsNotNone(request_meta.load_spec)
        self.assertEqual(request_meta.load_spec.kvpool_cached_tokens, 32)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_running_chunk_keeps_prefix_in_hbm_without_layer_reuse(self, mock_client_cls):
        config = self._make_config(
            extra_config={
                "backend": "memcache",
                "layerwise_num_shared_buffers": 4,
            },
            num_layers=4,
        )
        scheduler = KVPoolScheduler(config, use_layerwise=True)
        self._set_running_chunk(scheduler)

        meta = scheduler.build_connector_meta(self._make_running_chunk_output([1]))

        self.assertFalse(scheduler.layerwise_offload)
        self.assertEqual(len(meta.requests), 1)
        self.assertIsNone(meta.requests[0].load_spec)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_build_connector_meta_finished_req(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [0, 1])

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
    def test_build_connector_meta_preempted(self, mock_client_cls):
        config = self._make_config()
        scheduler = KVPoolScheduler(config, use_layerwise=False)

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import RequestTracker

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
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.MsgpackEncoder")
    def test_lookup(self, mock_encoder_cls, mock_zmq, mock_make_socket):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}

        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        mock_socket.recv.return_value = (32).to_bytes(4, "big")

        mock_encoder_cls.return_value.encode.side_effect = [[b"hashes"], [b"groups"]]
        client = LookupKeyClient(config)
        result = client.lookup(64, [b"\xaa\xbb"], hbm_hit_tokens=16)
        self.assertEqual(result, 32)
        mock_socket.send_multipart.assert_called_once()
        frames = mock_socket.send_multipart.call_args.args[0]
        self.assertEqual(
            frames,
            [
                (64).to_bytes(4, "big"),
                b"groups",
                (16).to_bytes(4, "big"),
                b"hashes",
            ],
        )

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


class TestKVPoolSchedulerStoreQueryKeys(unittest.TestCase):
    """Test _generate_store_query_keys method."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        return KVPoolScheduler(make_config(), use_layerwise=False)

    def test_generate_store_query_keys(self):
        cases = [
            ({}, False, 1),
            ({"num_layers": 4}, True, 4),
            ({"tp_size": 2, "put_step": 1}, False, 2),
        ]
        for attributes, include_layers, expected_count in cases:
            with self.subTest(attributes=attributes, include_layers=include_layers):
                scheduler = self._make_scheduler()
                for name, value in attributes.items():
                    setattr(scheduler, name, value)
                result = scheduler._generate_store_query_keys([b"\xaa\xbb"], include_layers=include_layers)
                self.assertEqual(len(result), 1)
                self.assertEqual(len(result[0]), expected_count)


class TestKVPoolSchedulerGetStoreLookupHitTokens(unittest.TestCase):
    """Test _get_store_lookup_hit_tokens method."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        return KVPoolScheduler(make_config(), use_layerwise=False)

    def test_store_lookup_hit_tokens(self):
        cases = [
            (4, [1, 1, 1, 1], 0, 64),
            (4, [1, 0, 0, 0], 0, 16),
            (4, [0, 0, 0, 0], 0, 0),
            (0, [], 0, 0),
            (4, [1, 1], 32, 64),
        ]
        for hash_count, exists, computed_tokens, expected in cases:
            with self.subTest(hash_count=hash_count, exists=exists, computed_tokens=computed_tokens):
                scheduler = self._make_scheduler()
                scheduler.store_scheduler.batch_is_exist.return_value = exists
                request = MagicMock()
                request.block_hashes = [b"\xaa"] * hash_count
                result = scheduler._get_store_lookup_hit_tokens(request, 64, computed_tokens)
                self.assertEqual(result, expected)


class TestKVPoolSchedulerFloorGranularity(unittest.TestCase):
    """Test _floor_to_cache_transfer_granularity."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_floor(self, mock_client_cls):
        scheduler = KVPoolScheduler(make_config(), use_layerwise=False)
        scheduler.cache_transfer_granularity = 16
        self.assertEqual(scheduler._floor_to_cache_transfer_granularity(33), 32)
        self.assertEqual(scheduler._floor_to_cache_transfer_granularity(16), 16)
        self.assertEqual(scheduler._floor_to_cache_transfer_granularity(15), 0)


class TestKVPoolSchedulerGetSwClippedBlocks(unittest.TestCase):
    """Test get_sw_clipped_blocks."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_sw_clipped_blocks(self, mock_client_cls):
        cases = [
            (False, [0], [[1, 2, 3]], [[1, 2, 3]]),
            (True, [2], [[1, 2, 3, 4, 5]], [[4, 5]]),
            (False, [0], [], []),
        ]
        for use_hybrid, num_swa_blocks, blocks, expected in cases:
            with self.subTest(use_hybrid=use_hybrid, blocks=blocks):
                scheduler = KVPoolScheduler(make_config(), use_layerwise=False)
                scheduler.use_hybrid = use_hybrid
                scheduler.num_swa_blocks = num_swa_blocks
                self.assertEqual(scheduler.get_sw_clipped_blocks(blocks), expected)


class TestKVPoolSchedulerUpdateFinished(unittest.TestCase):
    """Test update_finished_sending and update_finished_recving."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        return KVPoolScheduler(make_config(), use_layerwise=False)

    def test_update_finished(self):
        cases = [
            ("sending", {"r1", "r2", "r3"}, {"r1", "r2"}, {"r3"}),
            ("sending", {"r1"}, None, {"r1"}),
            ("recving", {"r1", "r2"}, {"r1"}, {"r2"}),
            ("recving", {"r1"}, None, {"r1"}),
        ]
        for direction, initial, finished, expected in cases:
            with self.subTest(direction=direction, finished=finished):
                scheduler = self._make_scheduler()
                attribute = "_delayed_free_req_ids" if direction == "sending" else "_loading_req_ids"
                setattr(scheduler, attribute, initial)
                getattr(scheduler, f"update_finished_{direction}")(finished)
                self.assertEqual(getattr(scheduler, attribute), expected)


class TestKVPoolSchedulerUpdateConnectorOutput(unittest.TestCase):
    """Test update_connector_output."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        config = make_config()
        config.parallel_config.world_size = 2
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        scheduler._block_pool = MagicMock()
        return scheduler

    def test_completed_event_frees_blocks(self):
        scheduler = self._make_scheduler()
        scheduler.sending_events = {1: 1}  # already 1 worker completed
        scheduler.sending_blocks = {1: [10, 20, 30]}
        scheduler._expected_worker_count = 2

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
            AscendStoreKVConnectorWorkerMetadata,
        )

        for initial, should_free in [(1, True), (0, False)]:
            with self.subTest(initial=initial):
                scheduler = self._make_scheduler()
                scheduler.sending_events = {1: initial}
                scheduler.sending_blocks = {1: [10, 20]}
                scheduler._expected_worker_count = 2
                output = MagicMock(kv_connector_worker_meta=AscendStoreKVConnectorWorkerMetadata({1: 1}))
                scheduler.update_connector_output(output)
                self.assertEqual(scheduler._block_pool.free_blocks.called, should_free)
                self.assertEqual(1 in scheduler.sending_blocks, not should_free)

    def test_invalid_event_id(self):
        scheduler = self._make_scheduler()
        scheduler.sending_events = {}
        scheduler.sending_blocks = {}

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
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
        scheduler = KVPoolScheduler(make_config(kv_role), use_layerwise=False)
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
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import RequestTracker

        for request_id, add_tracker, expected in [("missing", False, True), ("r1", True, False)]:
            with self.subTest(request_id=request_id):
                scheduler = self._make_scheduler()
                if add_tracker:
                    scheduler._request_trackers["r1"] = RequestTracker(
                        "r1", 32, allocated_block_ids=[0, 1], num_saved_tokens=0
                    )
                request = MagicMock(request_id=request_id)
                delay, _ = scheduler.request_finished_all_groups(request, ([1, 2],))
                self.assertEqual(delay, expected)

    def test_delay_free_with_blocks(self):
        scheduler = self._make_scheduler()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import RequestTracker

        scheduler._request_trackers["r1"] = RequestTracker("r1", 32, allocated_block_ids=[0, 1], num_saved_tokens=32)
        request = MagicMock()
        request.request_id = "r1"
        delay, _ = scheduler.request_finished_all_groups(request, ([1, 2],))
        self.assertTrue(delay)
        self.assertIn("r1", scheduler._delayed_free_req_ids)

    def test_no_delay_empty_blocks(self):
        scheduler = self._make_scheduler()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import RequestTracker

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


class TestKVPoolSchedulerGetLayerwiseHitTokens(unittest.TestCase):
    """Test _get_layerwise_hit_tokens."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls):
        # memcache backend makes the constructor resolve the real protocol
        # module; use_layerwise stays False so the test keeps exercising
        # the query_start_block offset math it was built around.
        return KVPoolScheduler(make_config(extra_config={"backend": "memcache"}), use_layerwise=False)

    def test_layerwise_hit_tokens(self):
        cases = [
            (2, [True, True], 32, 0, 32),
            (2, [True, False], 32, 0, 16),
            (1, [False], 16, 0, 0),
            (4, [True, True], 64, 32, 64),
        ]
        for hash_count, hits, token_count, computed_tokens, expected in cases:
            with self.subTest(hits=hits, computed_tokens=computed_tokens):
                scheduler = self._make_scheduler()
                key_infos = []
                for hit in hits:
                    info = MagicMock()
                    info.size.return_value = int(hit)
                    if hit:
                        info.gva_list.return_value = [0x1000]
                    key_infos.append(info)
                scheduler.store_scheduler.batch_get_key_info.return_value = key_infos
                request = MagicMock()
                request.block_hashes = [b"\xaa"] * hash_count
                result = scheduler._get_layerwise_hit_tokens(request, token_count, computed_tokens)
                self.assertEqual(result, expected)


class TestKVPoolSchedulerUpdateStateAfterAllocBranches(unittest.TestCase):
    """Test update_state_after_alloc additional branches."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls, extra_config=None):
        return KVPoolScheduler(make_config(extra_config=extra_config), use_layerwise=False)

    def test_async_adds_loading_req(self):
        scheduler = self._make_scheduler(extra_config={"load_async": True})
        scheduler.load_specs["r1"] = LoadSpec(0, 32, can_load=True)

        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1]]
        scheduler.update_state_after_alloc(request, blocks, 32)
        self.assertIn("r1", scheduler._loading_req_ids)

    def test_zero_external_tokens(self):
        scheduler = self._make_scheduler()
        scheduler.update_state_after_alloc(MagicMock(request_id="missing"), MagicMock(), 0)
        self.assertNotIn("missing", scheduler.load_specs)

        scheduler.use_layerwise = True
        scheduler.load_specs["r1"] = LoadSpec(0, 32, can_load=False)
        scheduler.update_state_after_alloc(MagicMock(request_id="r1"), MagicMock(), 0)
        self.assertTrue(scheduler.load_specs["r1"].can_load)


if __name__ == "__main__":
    unittest.main()
