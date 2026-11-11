# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import selective_sdma
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.selective_lim import get_selective_lim_library


class TestSelectiveLIMLibrary(unittest.TestCase):
    def test_missing_library_fails_instead_of_using_tensor_fallback(self):
        get_selective_lim_library.cache_clear()
        try:
            with (
                patch.dict(os.environ, {"VLLM_ASCEND_SELECTIVE_LIM_LIBRARY": ""}),
                self.assertRaisesRegex(RuntimeError, "VLLM_ASCEND_SELECTIVE_LIM_LIBRARY"),
            ):
                get_selective_lim_library()
        finally:
            get_selective_lim_library.cache_clear()


class TestSharedSDMAPrefetch(unittest.TestCase):
    def test_dynamic_batch_reads_shared_kv_and_acknowledges_after_dma(self):
        transfer = selective_sdma.SelectiveSDMATransfer.__new__(selective_sdma.SelectiveSDMATransfer)
        shared = torch.zeros(4 * (128 + 64), dtype=torch.uint8)
        transfer.coordinator = SimpleNamespace(participant=selective_sdma._GatherParticipant())
        transfer.host = [shared[:512].view(4, 128), shared[512:].view(4, 64)]
        transfer.device = [torch.empty_like(tensor) for tensor in transfer.host]
        transfer.sources = torch.empty(8, dtype=torch.int64)
        transfer.destinations = torch.empty_like(transfer.sources)
        transfer.lengths = torch.empty(8, dtype=torch.int32)
        transfer.addresses = [torch.zeros(4, dtype=torch.int64) for _ in range(2)]
        transfer.packed_addresses = torch.empty(8, dtype=torch.int64)
        transfer.payloads = {}
        operations = []

        for rows in (1, 4, 2):
            with self.subTest(rows=rows):
                operations.clear()
                transfer.copy = lambda *args: None
                transfer._launch = lambda key: operations.append("gather")
                transfer.copy_bytes = lambda dst, src, size, kind: operations.append((src, size, kind))

                def complete(stream, participant):
                    self.assertEqual(
                        ctypes.addressof(participant._obj), ctypes.addressof(transfer.coordinator.participant)
                    )
                    operations.append("ack")
                    return 0

                transfer.complete = complete
                manager = SimpleNamespace(copy_sfa_kv=lambda *args: operations.append("scatter"))
                addresses = torch.zeros(2 * rows, dtype=torch.int64)
                lengths = torch.zeros(2 * rows, dtype=torch.int32)
                with patch.object(
                    selective_sdma.torch_npu.npu, "current_stream", return_value=SimpleNamespace(npu_stream=1)
                ):
                    transfer.transfer(manager, addresses, addresses, lengths, None, to_device=True)
                self.assertEqual(
                    operations,
                    [
                        "gather",
                        (shared.data_ptr(), rows * 128, 1),
                        (shared.data_ptr() + rows * 128, rows * 64, 1),
                        "ack",
                        "scatter",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
