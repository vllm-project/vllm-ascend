# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import struct
import unittest
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    HostGatherCoordinator,
    SelectiveSDMATransfer,
)


class TestSharedSDMAPrefetch(unittest.TestCase):
    def test_coordinator_control_shares_header_and_keeps_byte_payload_offset(self):
        def create_mapping(*, create, size):
            return SimpleNamespace(name="selective-control-test", buf=bytearray(size), unlink=Mock())

        group = Mock()
        group.broadcast_object.side_effect = lambda name, src: name
        epoch = torch.zeros(1, dtype=torch.int64)
        with (
            patch.object(shared_memory, "SharedMemory", side_effect=create_mapping),
            patch.object(torch, "zeros", return_value=epoch),
            patch.object(torch.ops._C_ascend, "register_pinned_host_mmap", create=True) as register,
            patch.object(torch.ops._C_ascend, "unregister_pinned_host_mmap", create=True) as unregister,
        ):
            coordinator = HostGatherCoordinator(group, rank=0, world_size=4, capacity_bytes=64)
            try:
                self.assertEqual(coordinator.tensor.dtype, torch.uint8)
                self.assertEqual(coordinator.control.dtype, torch.int64)
                self.assertEqual(coordinator.control.data_ptr(), coordinator.tensor.data_ptr())
                coordinator.control[0] = 23
                coordinator.control[5] = 1
                self.assertEqual(struct.unpack_from("=q", coordinator.mapping.buf)[0], 23)
                self.assertEqual(struct.unpack_from("=q", coordinator.mapping.buf, 5 * 8)[0], 1)
                struct.pack_into("=q", coordinator.mapping.buf, 8, 97)
                self.assertEqual(coordinator.control[1].item(), 97)
                header_bytes = coordinator.control.numel() * coordinator.control.element_size()
                self.assertEqual(coordinator.payload.dtype, torch.uint8)
                self.assertEqual(coordinator.payload.numel(), 64)
                self.assertEqual(coordinator.payload.data_ptr(), coordinator.tensor.data_ptr() + header_bytes)
                coordinator.payload[0] = 0xA5
                self.assertEqual(coordinator.mapping.buf[header_bytes], 0xA5)
                register.assert_called_once_with(coordinator.tensor)
            finally:
                coordinator.registration()
            unregister.assert_called_once_with(coordinator.tensor)

    def test_dynamic_batch_reads_shared_kv_and_acknowledges_after_dma(self):
        transfer = SelectiveSDMATransfer.__new__(SelectiveSDMATransfer)
        shared = torch.zeros(4 * (128 + 64), dtype=torch.uint8)
        transfer.coordinator = SimpleNamespace(
            control=torch.zeros(10, dtype=torch.int64),
            epoch=torch.zeros(1, dtype=torch.int64),
            rank=1,
            world_size=4,
            payload=shared,
        )
        transfer.host = [shared[:512].view(4, 128), shared[512:].view(4, 64)]
        transfer.device = [torch.empty_like(tensor) for tensor in transfer.host]
        transfer.sources = torch.empty(8, dtype=torch.int64)
        transfer.destinations = torch.empty_like(transfer.sources)
        transfer.lengths = torch.empty(8, dtype=torch.int32)
        transfer.addresses = [torch.zeros(4, dtype=torch.int64) for _ in range(2)]
        transfer.packed_addresses = torch.empty(8, dtype=torch.int64)
        transfer.payloads = {}
        operations: list[str | tuple[int, int, int]] = []

        for rows in (1, 4, 2):
            with (
                self.subTest(rows=rows),
                patch.object(transfer, "copy", side_effect=lambda destination, source, kind: destination.copy_(source)),
                patch.object(
                    transfer, "enqueue", side_effect=lambda *args: operations.append("gather"), create=True
                ) as enqueue,
                patch.object(
                    transfer, "complete", side_effect=lambda *args: operations.append("ack"), create=True
                ) as complete,
                patch.object(
                    transfer,
                    "copy_bytes",
                    side_effect=lambda dst, src, size, kind: operations.append((src, size, kind)),
                ),
            ):
                operations.clear()
                manager = SimpleNamespace(copy_sfa_kv=lambda *args: operations.append("scatter"))
                addresses = torch.zeros(2 * rows, dtype=torch.int64)
                lengths = torch.tensor([128] * rows + [64] * rows, dtype=torch.int32)
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
                host_k, host_v = transfer.payloads[(rows, True)]
                self.assertEqual(host_k.shape, (rows, 128))
                self.assertEqual(host_v.shape, (rows, 64))
                self.assertEqual(host_k.data_ptr(), shared.data_ptr())
                self.assertEqual(host_v.data_ptr(), shared.data_ptr() + rows * 128)
                gather = enqueue.call_args.args
                self.assertIs(gather[0], transfer.device[0])
                self.assertIs(gather[4], host_k)
                self.assertIs(gather[5], host_v)
                self.assertIs(gather[6], transfer.coordinator.control)
                self.assertIs(gather[7], transfer.coordinator.epoch)
                self.assertEqual(gather[8:], (rows, 128, 64, 1, 4, True))
                complete.assert_called_once_with(
                    transfer.device[0],
                    transfer.coordinator.control,
                    transfer.coordinator.epoch,
                    transfer.coordinator.rank,
                )


if __name__ == "__main__":
    unittest.main()
