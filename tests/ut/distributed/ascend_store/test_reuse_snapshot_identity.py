# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta, RequestTracker


class TestReuseSnapshotIdentity(unittest.TestCase):
    def test_rollback_uses_saved_identity(self):
        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker.layerwise_offload = True
        worker.grouped_block_size = [128]
        worker.group_uses_align_state = [False]
        worker._reused_layers_by_group = {0: {1}}
        worker._snapshot_keys = {"request": {0: {0: (7, "snapshot:7")}}}
        request = ReqMeta("request", target_token_len=7, load_spec=LoadSpec(6, 6, True, layerwise_previous_step=True))
        _, get_key = worker._get_layerwise_load_snapshot(request, 0, 6, [])
        self.assertEqual(get_key(0, True), ("snapshot:7", False))
        worker._snapshot_keys["request"][0][0] = (128, "snapshot:128")
        _, get_key = worker._get_layerwise_load_snapshot(request, 0, 128, [])
        self.assertEqual(get_key(0, False), ("snapshot:128", False))
        worker._snapshot_keys["request"][0] = {1: (129, "snapshot:129")}
        _, get_key = worker._get_layerwise_load_snapshot(request, 0, 128, [])
        self.assertIsNone(get_key(0, False))

    def test_no_reuse_does_not_record_snapshot(self):
        tracker = RequestTracker("request", 129, allocated_block_ids_by_group=[[0, 1]])
        meta = ReqMeta.from_request_tracker(tracker, 128, block_hashes=["hash"], save_partial_block=False)
        self.assertIsNone(tracker.layerwise_recompute_start_token)
        self.assertIsNone(meta.layerwise_recompute_start_token)
