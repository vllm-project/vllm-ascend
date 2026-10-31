import os
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.utils import StatelessProcessGroup

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


class MockHcclLib:
    pass


class MockUniqueId:
    pass


class TestPyHcclCommunicator(TestBase):
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
    def test_world_size_1_return_early(self):
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 1, None, None),
            device="npu:0",
        )
        self.assertTrue(comm.disabled)
        self.assertFalse(comm.available)

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_load_hccl_fail(self):
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 2, None, None), device="npu:0", library_path="/not/exist/path/libhccl.so"
        )
        self.assertTrue(comm.disabled)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary", MockHcclLib)
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId", MockUniqueId)
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream", return_value=MagicMock(npu_stream=5678))
    def test_stateless_group(self, *_):
        group = StatelessProcessGroup(rank=3, world_size=4, store=None)

        comm = PyHcclCommunicator(group=group, device=3)

        self.assertEqual(comm.rank, 3)
        self.assertEqual(comm.world_size, 4)

    @patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"})
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary", MockHcclLib)
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId", MockUniqueId)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="nccl")
    @patch("torch.distributed.Backend.HCCL", "hccl", create=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.distributed.broadcast")
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream", return_value=MagicMock(npu_stream=1234))
    def test_multi_gpu_pg_torch(
        self,
        *_,
    ):
        fake_pg = MagicMock()
        comm = PyHcclCommunicator(group=fake_pg, device="npu:1")

        self.assertEqual(comm.rank, 1)
        self.assertEqual(comm.world_size, 2)
        self.assertFalse(comm.available)
        self.assertTrue(comm.disabled)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.device_communicators.pyhccl.torch.npu.device")
    def test_close_synchronizes_and_destroys_once(self, _mock_device, synchronize):
        comm = PyHcclCommunicator.__new__(PyHcclCommunicator)
        comm.available = True
        comm.disabled = False
        comm.device = "npu:0"
        comm.hccl = MagicMock()
        comm.comm = MagicMock()
        tensor = MagicMock(device=comm.device, dtype=torch.float32)
        tensor.data_ptr.return_value = 1
        tensor.numel.return_value = 1
        comm.broadcast(tensor, src=0, stream=MagicMock(npu_stream=1234))
        calls = MagicMock()
        calls.attach_mock(synchronize, "synchronize")
        calls.attach_mock(comm.hccl.hcclCommDestroy, "destroy")

        comm.close()
        comm.close()

        synchronize.assert_called_once_with(comm.device)
        comm.hccl.hcclCommDestroy.assert_called_once_with(comm.comm)
        self.assertEqual([call[0] for call in calls.mock_calls], ["synchronize", "destroy"])
        self.assertFalse(comm.available)
        for collective in (lambda: comm.broadcast(tensor, src=0), lambda: comm.all_reduce(tensor)):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                collective()

    def test_close_disabled_communicator(self):
        comm = PyHcclCommunicator(group=StatelessProcessGroup(0, 1, None, None), device="npu:0")
        comm.close()
        comm.close()
        self.assertTrue(comm.disabled)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.device_communicators.pyhccl.torch.npu.device")
    def test_close_failure_propagates_without_retry(self, _mock_device, synchronize):
        for operation in ("synchronize", "destroy"):
            with self.subTest(operation=operation):
                synchronize.reset_mock(side_effect=True)
                comm = PyHcclCommunicator.__new__(PyHcclCommunicator)
                comm.available = True
                comm.device = "npu:0"
                comm.hccl = MagicMock()
                comm.comm = MagicMock()
                failing_call = synchronize if operation == "synchronize" else comm.hccl.hcclCommDestroy
                failing_call.side_effect = RuntimeError("close failed")

                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    comm.close()
                comm.close()

                synchronize.assert_called_once_with(comm.device)
                if operation == "synchronize":
                    comm.hccl.hcclCommDestroy.assert_not_called()
                else:
                    comm.hccl.hcclCommDestroy.assert_called_once_with(comm.comm)
                self.assertFalse(comm.available)
