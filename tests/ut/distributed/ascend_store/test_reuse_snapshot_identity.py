# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta


class TestReuseSnapshotIdentity(unittest.TestCase):
    def test_previous_step_uses_public_hash_and_request_local_tail_key(self):
        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker.grouped_block_size = [128, 128]
        worker._reused_layers_by_group = {0: {1}, 1: {1}}
        worker._snapshot_keys = {"request": {0: {0: "old-private", 1: "tail"}, 1: {1: "other-group"}}}
        worker._make_layerwise_full_key = lambda group, block_hash: f"full:{group}:{block_hash}"
        request = ReqMeta("request", target_token_len=131)
        snapshot = worker._get_layerwise_load_snapshot(request, 0, 129, [b"\xab"])
        assert snapshot is not None
        full_blocks, get_saved_key = snapshot
        self.assertEqual(full_blocks, 1)
        self.assertEqual(get_saved_key(0), ("full:0:ab", True))
        self.assertEqual(get_saved_key(1), ("tail", False))
        other_group = worker._get_layerwise_load_snapshot(request, 1, 129, [])
        assert other_group is not None
        self.assertEqual(other_group[1](1), ("other-group", False))
        other_request = worker._get_layerwise_load_snapshot(ReqMeta("other"), 0, 129, [])
        assert other_request is not None
        self.assertIsNone(other_request[1](1))

    def test_resident_group_does_not_load_a_snapshot(self):
        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker._reused_layers_by_group = {1: {0}}
        request = ReqMeta("request", target_token_len=131)
        self.assertIsNone(worker._get_layerwise_load_snapshot(request, 0, 129, []))
