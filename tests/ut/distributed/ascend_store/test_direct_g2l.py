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

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    LayerBlockRange,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)

DIRECT_G2L_CONFIG = {"backend": "memcache"}


@pytest.fixture(autouse=True)
def _patch_pool_scheduler_importlib():
    """Point the scheduler's dynamic backend import at a MagicMock so
    ``store_scheduler`` is a mock (same pattern as test_pool_scheduler.py)."""
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as mock_importlib:
        mock_importlib.import_module.return_value = MagicMock()
        yield


def make_key_info(gva=0x1000):
    info = MagicMock()
    hit = gva > 0
    info.size.return_value = int(hit)
    info.gva_list.return_value = [gva]
    return info


def make_scheduler_config(kv_role="kv_both", extra_config=None, block_size=16):
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


class TestSchedulerDirectG2L(unittest.TestCase):
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def _make_scheduler(self, mock_client_cls, extra_config=None):
        scheduler = KVPoolScheduler(
            make_scheduler_config(extra_config=extra_config or DIRECT_G2L_CONFIG),
            use_layerwise=True,
        )
        scheduler.store_scheduler.batch_add_lease.side_effect = lambda keys, ttl: [0] * len(keys)
        return scheduler

    def _make_request(self, req_id="r1", hash_count=2):
        request = MagicMock()
        request.request_id = req_id
        request.block_hashes = [bytes([0xA0 + i]) for i in range(hash_count)]
        return request

    def test_hit_snapshot_and_pass_reuse(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_get_key_info.return_value = [make_key_info(0x1000), make_key_info(0x2000)]
        request = self._make_request()

        first = scheduler._get_layerwise_hit_tokens(request, 32, 0)
        self.assertEqual(first, 32)
        # one hit-check query + one snapshot query, one lease
        self.assertEqual(scheduler.store_scheduler.batch_get_key_info.call_count, 2)
        self.assertEqual(scheduler.store_scheduler.batch_add_lease.call_count, 1)

        second = scheduler._get_layerwise_hit_tokens(request, 32, 0)
        self.assertEqual(second, 32)
        # pass reuse: no new RPCs
        self.assertEqual(scheduler.store_scheduler.batch_get_key_info.call_count, 2)
        self.assertEqual(scheduler.store_scheduler.batch_add_lease.call_count, 1)

        snap = scheduler._pool_gva_snapshots["r1"]
        self.assertEqual(snap.gvas_by_group, [[0x1000, 0x2000]])
        self.assertEqual(len(snap.lease_keys), 2)

    def test_renewal_near_deadline(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_get_key_info.return_value = [make_key_info(), make_key_info()]
        request = self._make_request()
        scheduler._get_layerwise_hit_tokens(request, 32, 0)

        snap = scheduler._pool_gva_snapshots["r1"]
        snap.deadline = time.time() + 1.0  # below the renew horizon
        scheduler._get_layerwise_hit_tokens(request, 32, 0)
        self.assertEqual(scheduler.store_scheduler.batch_add_lease.call_count, 2)
        self.assertGreater(snap.deadline, time.time() + 50.0)

    def test_lease_failure_zeros_gva(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_get_key_info.return_value = [make_key_info(), make_key_info()]
        scheduler.store_scheduler.batch_add_lease.side_effect = lambda keys, ttl: [1] * len(keys)
        scheduler._get_layerwise_hit_tokens(self._make_request(), 32, 0)

        snap = scheduler._pool_gva_snapshots["r1"]
        self.assertEqual(snap.gvas_by_group, [[0, 0]])
        self.assertEqual(snap.lease_keys, [])

    def test_shared_keys_release_on_last_dep(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_get_key_info.return_value = [make_key_info(), make_key_info()]
        scheduler._get_layerwise_hit_tokens(self._make_request("r1"), 32, 0)
        scheduler._get_layerwise_hit_tokens(self._make_request("r2"), 32, 0)
        self.assertEqual(scheduler.store_scheduler.batch_get_key_info.call_count, 4)

        scheduler._release_pool_lease("r1")
        scheduler.store_scheduler.batch_remove_lease.assert_not_called()

        scheduler._release_pool_lease("r2")
        scheduler.store_scheduler.batch_remove_lease.assert_called_once()
        released = scheduler.store_scheduler.batch_remove_lease.call_args.args[0]
        self.assertEqual(len(released), 2)
        self.assertFalse(scheduler._pool_leases)

    def test_update_connector_output_releases_loaded_req(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_get_key_info.return_value = [make_key_info(), make_key_info()]
        scheduler._get_layerwise_hit_tokens(self._make_request("r1"), 32, 0)

        output = MagicMock()
        output.kv_connector_worker_meta = AscendStoreKVConnectorWorkerMetadata(loaded_req_ids=["r1"])
        scheduler.update_connector_output(output)
        scheduler.store_scheduler.batch_remove_lease.assert_called_once()

    def test_attach_direct_g2l_to_meta(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_get_key_info.return_value = [make_key_info(), make_key_info()]
        scheduler._get_layerwise_hit_tokens(self._make_request("r1"), 32, 0)

        meta = AscendConnectorMetadata(set())
        req = ReqMeta(req_id="r1", token_len_chunk=32, block_ids=[[7, 8]], block_hashes=[b"\xaa"] * 2)
        req.load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True)
        meta.add_request(req)

        scheduler._attach_direct_g2l(meta)
        self.assertEqual(req.pool_load_gvas_by_group, [[0x1000, 0x1000]])
        self.assertIsNotNone(req.pool_lease_deadline)

    def test_attach_rebuilds_for_larger_range(self):
        scheduler = self._make_scheduler()
        scheduler.store_scheduler.batch_get_key_info.return_value = [make_key_info(), make_key_info()]
        scheduler._get_layerwise_hit_tokens(self._make_request("r1"), 32, 0)

        meta = AscendConnectorMetadata(set())
        req = ReqMeta(req_id="r1", token_len_chunk=48, block_ids=[[7, 8, 9]], block_hashes=[b"\xaa"] * 3)
        # offload-style spec: needs more tokens than the snapshot covers
        req.load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=48, can_load=True)
        meta.add_request(req)

        scheduler._attach_direct_g2l(meta)
        # one extra snapshot build for the wider range
        self.assertEqual(scheduler.store_scheduler.batch_get_key_info.call_count, 3)
        self.assertEqual(len(req.pool_load_gvas_by_group[0]), 3)


class TestWorkerDirectG2L(unittest.TestCase):
    def _make_worker(self):
        worker = make_worker(
            self,
            kv_role="kv_both",
            extra_config={"backend": "memcache"},
            use_layerwise=True,
        )
        self.assertTrue(worker.use_layerwise_transfer)
        return worker

    def _make_load_request(self, deadline=None, gvas=None):
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[[7, 8]],
            block_hashes=[b"\xaa", b"\xbb"],
            block_ids_by_group_np=[np.asarray([7, 8], dtype=np.int64)],
        )
        request.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=32,
            can_load=True,
            kvpool_store_skip_tokens=32,
        )
        request.pool_load_gvas_by_group = gvas
        request.pool_lease_deadline = deadline if deadline is not None else time.time() + 60.0
        return request

    def test_table_lookup_without_rpc(self):
        worker = self._make_worker()
        request = self._make_load_request(gvas=[[0x1000, 0x2000]])
        worker._prepare_load_gvas([request])
        self.assertEqual(request.load_block_gvas_np.tolist(), [0x1000, 0x2000])
        worker.m_store.batch_get_key_info.assert_not_called()
        worker.m_store.batch_add_lease.assert_not_called()
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_zero_gva_reports_invalid_blocks(self):
        worker = self._make_worker()
        request = self._make_load_request(gvas=[[0x1000, 0]])
        worker._prepare_load_gvas([request])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {8})

    def test_expired_deadline_treated_as_miss(self):
        worker = self._make_worker()
        request = self._make_load_request(deadline=time.time(), gvas=[[0x1000, 0x2000]])
        worker._prepare_load_gvas([request])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7, 8})

    def test_loaded_req_reporting(self):
        worker = self._make_worker()
        worker._on_direct_g2l_req_loaded("r1")
        meta = worker.build_connector_worker_meta()
        self.assertEqual(meta.loaded_req_ids, ["r1"])
        # drained after one report
        self.assertIsNone(worker.build_connector_worker_meta())


class TestLayerRecvingThreadDirectG2L(unittest.TestCase):
    def _make_thread(self, load_failure_cb=None):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread.load_failure_cb = load_failure_cb
        thread.loaded_req_cb = None
        thread.step_lease_deadline = None
        return thread

    def _make_task(self, block_ids=(7, 8)):
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[list(block_ids)],
            block_hashes=[b"\xaa"],
            block_ids_by_group_np=[np.asarray(block_ids, dtype=np.int64)],
        )
        block_range = LayerBlockRange(request=request, start_block=0, end_block=2)
        return LayerTransferTask(layer_id=0, block_ranges=[block_range], group_id=0)

    def test_expiry_reports_all_blocks(self):
        failed: list[set] = []
        thread = self._make_thread(load_failure_cb=lambda ids: failed.append(ids))
        thread._report_direct_g2l_expiry([self._make_task()])
        self.assertEqual(failed, [{7, 8}])

    def test_batch_copy_flag_only_for_g2l(self):
        thread = self._make_thread()
        thread.m_store = MagicMock()
        thread.num_addrs_per_block = 1
        gvas = np.asarray([0x1000], dtype=np.int64)
        addrs = np.asarray([100], dtype=np.int64)
        sizes = np.asarray([8], dtype=np.int64)

        thread._batch_copy_with_limits(gvas, addrs, sizes, 1, 0, 0, skip_validation=True)
        args = thread.m_store.store.batch_copy.call_args.args
        self.assertEqual(args, ([0x1000], [100], [8], 1, 1))

        thread.m_store.store.batch_copy.reset_mock()
        thread._batch_copy_with_limits(gvas, addrs, sizes, 0, 0, 0, skip_validation=True)
        args = thread.m_store.store.batch_copy.call_args.args
        self.assertEqual(args, ([0x1000], [100], [8], 0))


if __name__ == "__main__":
    unittest.main()
