# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker as module


class TestReusePublicationFence(unittest.TestCase):
    def test_fence_order_and_scope(self):
        for transfer, reuse, tp_size, sync in [
            (True, True, 16, True),
            (True, True, 1, False),
            (True, False, 16, False),
            (False, True, 16, False),
        ]:
            for requests in ([], [object()]):
                with self.subTest(transfer=transfer, reuse=reuse, tp_size=tp_size, empty=not requests):
                    worker = module.KVPoolWorker.__new__(module.KVPoolWorker)
                    worker.use_layerwise = True
                    worker.use_layerwise_transfer = transfer
                    worker.layerwise_offload = reuse
                    worker.tp_size = tp_size
                    worker._save_disabled = threading.Event()
                    worker.kv_recv_thread = None
                    calls: list[str] = []
                    worker._drain_deferred_last_save = lambda calls=calls: calls.append("drain")
                    worker.process_layer_data = lambda requests, calls=calls: calls.append("prepare")
                    group = SimpleNamespace(barrier=lambda calls=calls: calls.append("barrier"))
                    with (
                        patch.object(module, "get_tp_group", return_value=group),
                        patch.object(module, "reset_attention_compute_start_gate"),
                    ):
                        worker.prepare_layerwise_step(SimpleNamespace(requests=requests))
                    self.assertEqual(calls, ["drain", "barrier", "prepare"] if sync else ["drain", "prepare"])

    def test_failed_local_save_does_not_query(self):
        worker = module.KVPoolWorker.__new__(module.KVPoolWorker)
        worker.use_layerwise = True
        with (
            patch.object(worker, "_drain_deferred_last_save", side_effect=RuntimeError("save failed")),
            patch.object(worker, "process_layer_data") as prepare,
            patch.object(module, "get_tp_group") as group,
            self.assertRaisesRegex(RuntimeError, "save failed"),
        ):
            worker.prepare_layerwise_step(SimpleNamespace(requests=[]))
        prepare.assert_not_called()
        group.assert_not_called()
