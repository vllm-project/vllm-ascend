# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker as module


class TestReusePublicationFence(unittest.TestCase):
    def test_wait_for_previous_save_drains_once(self):
        worker = module.LayerwiseKVPoolWorker.__new__(module.LayerwiseKVPoolWorker)
        worker._pending_last_save_drain = True
        worker.kv_send_thread = object()
        worker.num_layers = 2

        with patch.object(worker, "_wait_for_final_layer_save") as wait:
            worker.wait_for_previous_save()
            worker.wait_for_previous_save()

        wait.assert_called_once_with(2)
        self.assertFalse(worker._pending_last_save_drain)

    def test_fence_order_and_scope(self):
        for transfer, reuse, tp_size, pcp_size, barriers in [
            (True, True, 16, 4, ["tp", "pcp"]),
            (True, True, 1, 4, ["pcp"]),
            (True, True, 16, 1, ["tp"]),
            (True, True, 1, 1, []),
            (True, False, 16, 4, []),
            (False, True, 16, 4, []),
        ]:
            for requests in ([], [object()]):
                with self.subTest(
                    transfer=transfer, reuse=reuse, tp_size=tp_size, pcp_size=pcp_size, empty=not requests
                ):
                    worker = module.LayerwiseKVPoolWorker.__new__(module.LayerwiseKVPoolWorker)
                    worker.use_layerwise = True
                    worker.use_kvpp = False
                    worker.use_layerwise_transfer = transfer
                    worker.layerwise_offload = reuse
                    worker.pcp_size = pcp_size
                    worker.tp_size = tp_size
                    worker._save_disabled = threading.Event()
                    worker.kv_recv_thread = None
                    calls: list[str] = []
                    tp_group = SimpleNamespace(barrier=lambda calls=calls: calls.append("tp"))
                    pcp_group = SimpleNamespace(barrier=lambda calls=calls: calls.append("pcp"))
                    with (
                        patch.object(
                            worker, "_drain_deferred_last_save", side_effect=lambda calls=calls: calls.append("drain")
                        ),
                        patch.object(
                            worker, "process_layer_data", side_effect=lambda _, calls=calls: calls.append("prepare")
                        ),
                        patch.object(module, "get_tp_group", return_value=tp_group),
                        patch.object(module, "get_pcp_group", return_value=pcp_group),
                        patch.object(module, "reset_attention_gates"),
                    ):
                        worker.prepare_layerwise_step(SimpleNamespace(requests=requests))
                    self.assertEqual(calls, ["drain", *barriers, "prepare"])

    def test_failed_local_save_does_not_query(self):
        worker = module.LayerwiseKVPoolWorker.__new__(module.LayerwiseKVPoolWorker)
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
