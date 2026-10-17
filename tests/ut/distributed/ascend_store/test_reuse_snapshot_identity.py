# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta, RequestTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class TestReuseSnapshotIdentity(unittest.TestCase):
    def test_rollback_uses_saved_identity(self):
        tracker = RequestTracker("request", 7, allocated_block_ids_by_group=[[0]])
        meta = ReqMeta.from_request_tracker(tracker, 128, block_hashes=[], save_partial_block=True)
        self.assertTrue(meta.can_save)
        self.assertEqual(tracker.last_layerwise_snapshot_end_token, 7)
        self.assertEqual(tracker.last_layerwise_snapshot_block_index, 0)
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.layerwise_offload = True
        worker.num_kv_cache_groups = 1
        worker.group_uses_align_state = [False]
        spec = LoadSpec(6, 6, True, layerwise_snapshot_end_token=7, layerwise_snapshot_block_index=0)
        request = ReqMeta("request", target_token_len=7, load_spec=spec)
        self.assertEqual(worker._resolve_layerwise_load_partial(request, 0, 6, 128, 0), (0, 7))
        spec.layerwise_snapshot_end_token = 128
        self.assertEqual(worker._resolve_layerwise_load_partial(request, 0, 128, 128, 1), (0, 128))
        spec.layerwise_snapshot_end_token = 129
        spec.layerwise_snapshot_block_index = 1
        self.assertEqual(worker._resolve_layerwise_load_partial(request, 0, 128, 128, 1), (None, 128))
        spec.layerwise_snapshot_end_token = None
        spec.layerwise_snapshot_block_index = None
        self.assertEqual(worker._resolve_layerwise_load_partial(request, 0, 6, 128, 0), (0, 6))

    def test_no_reuse_does_not_record_snapshot(self):
        tracker = RequestTracker("request", 129, allocated_block_ids_by_group=[[0, 1]])
        ReqMeta.from_request_tracker(tracker, 128, block_hashes=["hash"], save_partial_block=False)
        self.assertIsNone(tracker.last_layerwise_snapshot_end_token)
        self.assertIsNone(tracker.last_layerwise_snapshot_block_index)
